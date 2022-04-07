import os
import math
import time
import random
import rclpy # ROS Client Library for Python
import gym
from gym import spaces
import numpy as np

# Package Management Library
from ament_index_python.packages import get_package_share_directory 

# Messages
from std_srvs.srv import Empty
from geometry_msgs.msg import Pose
from geometry_msgs.msg import Twist # Twist is linear and angular velocity
from sensor_msgs.msg import LaserScan # LaserScan contains the value of the laser rays
from nav_msgs.msg import Odometry # Position, orientation, linear velocity, angular velocity

# Gazebo's services
from gazebo_msgs.srv import SpawnEntity 
from gazebo_msgs.srv import DeleteEntity 


class DroneEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['console']}

    def __init__(self, episode_max_steps = 300):
        super(DroneEnv, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects

        self.n_actions = 6
        self.n_states = 12

        # actions: Front, Back, Left, Right, Down, Up
        self.action_space = spaces.Discrete(self.n_actions)

        # bound parameters
        self.vertical_bound = 4.0       # bound from 0 to +val
        self.horizontal_bound = 5.0     # bound from -val to +val

        self.spawn_bound_half_side = 1.5
        self.spawn_vertical_min = 2.0
        self.spawn_vertical_max = 3.0

        # plugin parameters used for normalization
        self.platform_center = [0.0, 0.0, 0.2 + 0.1] # 0.1 represent half height 

        self.max_laser_angle = np.pi/12

        self.max_roll_pitch = np.pi/6 + 0.011 # this value correspond to a upperbound of the rotation step used in the plugin

        self.plugin_hover_vel = 0.294
        self.max_horizontal_vel = np.sin(self.max_roll_pitch) * self.plugin_hover_vel * 2
        self.max_vertical_vel = 0.055

        self.max_platform_rot = np.pi/12
        self.max_delta_roll_pitch = self.max_roll_pitch + self.max_platform_rot

        self.max_delta_vertical = self.vertical_bound - self.platform_center[2] + np.sin(self.max_platform_rot) * 1 # 1 represent half of the platform side
        # this value represent the max min value read by the lasers, it's used in certain cases to calculate delta_z
        self.max_laser_vertical_length = self.max_delta_vertical / np.cos(self.max_laser_angle)

        # states: roll, pitch, v_x, v_y, v_z, OUT_F, OUT_B, OUT_L, OUT_R, Delta_roll, Delta_pitch, Delta_z
        low =  [-1, -1, -1, -1, -1, 0, 0, 0, 0, -1, -1, 0]
        high = [ 1,  1,  1,  1,  1, 1, 1, 1, 1,  1,  1, 1]
        self.observation_space = spaces.Box(
            low=np.array(low), 
            high=np.array(high), 
            shape=(self.n_states,), 
            dtype=np.float64)

        # --- Node initialization -----------------------------------------------------------------
        rclpy.init()
        self.node = rclpy.create_node(self.__class__.__name__)

        # --- Gazebo collector stuff --------------------------------------------------------------
        # Get path to the quadrotor
        sdf_file_path = os.path.join(get_package_share_directory("drone_worker"), "models", "quadrotor_laser", "model.sdf")

        try:
            self.model = open(sdf_file_path, 'r').read()
        except IOError as e:
            self.node.get_logger().error('Error reading file {}: {}'.format(sdf_file_path, e))
            return

        self.quadcopter_namespace = "demo"
        self.quadcopter_name = "Quadcopter"

        self.skip_msg = 50
        self.skip_count_odom = 0
        self.skip_count_laser = 0

        self.last_odom_pos = [0] * 3
        self.last_odom_rot = [0] * 3
        self.last_laserscan_rays = [0] * 9

        self.odom_updated = False
        self.laser_updated = False

        self.platform_rot = [0.0, 0.0]

        self.step_count = 0
        self.run_max_steps = episode_max_steps

        self.done_flag = False
        self.reset_flag = False

        self.reward_multiplier = 10
        self.reward_penalty = 2 * self.run_max_steps

        # ROS stuff -------------------------------------------------------------------------------

        # Create a subscriber
        # This node subscribes to messages of type
        # nav_msgs/Odometry (i.e. position and orientation of the robot)
        self.odom_subscriber = self.node.create_subscription(Odometry,
                                                                '/' + self.quadcopter_namespace + '/odom',
                                                                self.odom_callback,
                                                                10)
        # Create a subscriber
        # This node subscribes to messages of type
        # sensor_msgs/LaserScan (i.e. values of laser rays)
        self.laser_subscriber = self.node.create_subscription(LaserScan,
                                                                '/' + self.quadcopter_namespace + '/laser',
                                                                self.laser_callback,
                                                                10)

        self.platform_subscriber = self.node.create_subscription(Odometry,
                                                                  '/platform_odom',
                                                                  self.platform_callback,
                                                                  10)
        
        # Create a publisher
        # This node publishes the desired linear and angular velocity of the robot (in the
        # robot chassis coordinate frame) to the /demo/cmd_vel topic. Using the diff_drive
        # plugin enables the robot model to read this /demo/cmd_vel topic and execute
        # the motion accordingly.
        self.velocity_publisher = self.node.create_publisher(Twist,
                                                                '/' + self.quadcopter_namespace + '/cmd_vel',
                                                                10)

        # Gazebo services clients -----------------------------------------------------------------
        service_timeout = 5.0

        self.client_spawn = self.node.create_client(SpawnEntity, '/spawn_entity')
        if not self.client_spawn.wait_for_service(timeout_sec=service_timeout):
            self.node.get_logger().error('Service /spawn_entity unavailable. Was Gazebo started with GazeboRosFactory?')

        self.client_delete = self.node.create_client(DeleteEntity, '/delete_entity')
        if not self.client_delete.wait_for_service(timeout_sec=service_timeout):
            self.node.get_logger().error('Service /delete_entity unavailable. Was Gazebo started with GazeboRosFactory?')

        self.client_unpause = self.node.create_client(Empty, '/unpause_physics')
        if not self.client_unpause.wait_for_service(timeout_sec=service_timeout):
            self.node.get_logger().error('Service /unpause_physics unavailable. Was Gazebo started right?')

        self.client_pause = self.node.create_client(Empty, '/pause_physics')
        if not self.client_pause.wait_for_service(timeout_sec=service_timeout):
            self.node.get_logger().error('Service /pause_physics unavailable. Was Gazebo started right?')

        self.client_resetsim = self.node.create_client(Empty, '/reset_simulation')
        if not self.client_resetsim.wait_for_service(timeout_sec=service_timeout):
            self.node.get_logger().error('Service /reset_simulation unavailable. Was Gazebo started right?')

        self.client_resetwrld = self.node.create_client(Empty, '/reset_world')
        if not self.client_resetwrld.wait_for_service(timeout_sec=service_timeout):
            self.node.get_logger().error('Service /reset_world unavailable. Was Gazebo started right?')

        # -----------------------------------------------------------------------------------------

    def step(self, action):
        self.step_count += 1

        self.execute_action(self.translate_action(action))
        #self.node.get_logger().info("Action: %s" % action)

        # get new observation state
        observation = self.wait_and_get_obs()

        # check if we reached the max number of step in a run
        if self.step_count >= self.run_max_steps:
            self.reset_flag = True

        # calculate the reward of the current observation state
        reward = self.calculate_reward()

        # calculate the done flag
        done = self.done_flag or self.reset_flag

        info = {}
        return observation, reward, done, info

    def reset(self):
        # Reset world and model position with a new random position
        initial_pose = Pose()

        initial_pose.position.x = random.uniform(-self.spawn_bound_half_side, self.spawn_bound_half_side)
        initial_pose.position.y = random.uniform(-self.spawn_bound_half_side, self.spawn_bound_half_side)
        initial_pose.position.z = random.uniform(self.spawn_vertical_min, self.spawn_vertical_max)

        # remove model if present and reset world and simulation
        self.pause_physics()
        self.delete_entity()
        time.sleep(0.2)
        self.reset_world()
        self.reset_sim()

        # spawn model in the new position and unpause physics
        self.spawn_entity(initial_pose)
        self.unpause_physics()

        # get new observation state
        observation = self.wait_and_get_obs()

        self.step_count = 0
        self.done_flag = False
        self.reset_flag = False

        return observation

    def render(self, mode='console'):
        print("Rendering not available.")

    def close(self):
        self.node.destroy_node()
        rclpy.shutdown()




    # ---------------------------------------------------------------------------------------------
    # --- Env Functions ---------------------------------------------------------------------------
    # ---------------------------------------------------------------------------------------------

    def wait_and_get_obs(self):
        # wait for callback new values
        while not self.are_values_updated():
            rclpy.spin_once(self.node)

        #self.print_odom_and_laser() # TODO: remove (only for testing)

        state = self.get_state()
        #self.node.get_logger().info("State: %s" % state)
        return state

    def get_state(self):
        # normalize roll and pitch values
        s = [self.last_odom_rot[0] / self.max_roll_pitch, 
             self.last_odom_rot[1] / self.max_roll_pitch]
        
        # normalize velocity values
        s += [self.last_odom_vel_linear[0] / self.max_horizontal_vel, 
              self.last_odom_vel_linear[1] / self.max_horizontal_vel, 
              np.clip(self.last_odom_vel_linear[2] / self.max_vertical_vel, -1, 1)]

        out_front = self.last_laserscan_rays[7] == math.inf
        out_back = self.last_laserscan_rays[1] == math.inf
        out_left = self.last_laserscan_rays[5] == math.inf
        out_right = self.last_laserscan_rays[3] == math.inf

        s += [int(out_front), int(out_back), int(out_left), int(out_right)]

        laser_angle = math.pi / 12

        delta_roll = 0
        if self.last_odom_rot[0] > 0: # inclined in the leftward direction
            if out_left + (self.last_laserscan_rays[4] == math.inf) + out_right > 1:
                delta_roll = self.max_delta_roll_pitch
            elif out_left:
                right_laser = self.last_laserscan_rays[4] / math.cos(laser_angle)
                remaining_right_laser = self.last_laserscan_rays[3] - right_laser
                laser_paral_dist = right_laser * math.sin(laser_angle)
                scalene_triang_known_angle = math.pi - laser_angle
                platform_side = math.sqrt(laser_paral_dist ** 2 + remaining_right_laser ** 2 - 2 * laser_paral_dist * remaining_right_laser * math.cos(scalene_triang_known_angle))
                delta_roll = math.asin(math.sin(scalene_triang_known_angle) * remaining_right_laser / platform_side)
            else:
                central_laser = self.last_laserscan_rays[5] * math.cos(laser_angle)
                laser_paral_dist = self.last_laserscan_rays[5] * math.sin(laser_angle)
                delta_roll = math.atan((self.last_laserscan_rays[4] - central_laser) / laser_paral_dist) 
        elif self.last_odom_rot[0] < 0: # inclined in the rightward direction
            if out_left + (self.last_laserscan_rays[4] == math.inf) + out_right > 1:
                delta_roll = -self.max_delta_roll_pitch
            elif out_right:
                left_laser = self.last_laserscan_rays[4] / math.cos(laser_angle)
                remaining_left_laser = self.last_laserscan_rays[5] - left_laser
                laser_paral_dist = left_laser * math.sin(laser_angle)
                scalene_triang_known_angle = math.pi - laser_angle
                platform_side = math.sqrt(laser_paral_dist ** 2 + remaining_left_laser ** 2 - 2 * laser_paral_dist * remaining_left_laser * math.cos(scalene_triang_known_angle))
                delta_roll = -math.asin(math.sin(scalene_triang_known_angle) * remaining_left_laser / platform_side)
            else:
                central_laser = self.last_laserscan_rays[3] * math.cos(laser_angle)
                laser_paral_dist = self.last_laserscan_rays[3] * math.sin(laser_angle)
                delta_roll = -math.atan((self.last_laserscan_rays[4] - central_laser) / laser_paral_dist) 

        delta_pitch = 0
        if self.last_odom_rot[1] > 0: # inclined in the forward direction
            if out_front + (self.last_laserscan_rays[4] == math.inf) + out_back > 1:
                delta_pitch = self.max_delta_roll_pitch
            elif out_front:
                back_laser = self.last_laserscan_rays[4] / math.cos(laser_angle)
                remaining_back_laser = self.last_laserscan_rays[1] - back_laser
                laser_paral_dist = back_laser * math.sin(laser_angle)
                scalene_triang_known_angle = math.pi - laser_angle
                platform_side = math.sqrt(laser_paral_dist ** 2 + remaining_back_laser ** 2 - 2 * laser_paral_dist * remaining_back_laser * math.cos(scalene_triang_known_angle))
                delta_pitch = math.asin(math.sin(scalene_triang_known_angle) * remaining_back_laser / platform_side)
            else:
                central_laser = self.last_laserscan_rays[7] * math.cos(laser_angle)
                laser_paral_dist = self.last_laserscan_rays[7] * math.sin(laser_angle)
                delta_pitch = math.atan((self.last_laserscan_rays[4] - central_laser) / laser_paral_dist) 
        elif self.last_odom_rot[1] < 0: # inclined in the backward direction
            if out_front + (self.last_laserscan_rays[4] == math.inf) + out_back > 1:
                delta_pitch = -self.max_delta_roll_pitch
            elif out_back:
                front_laser = self.last_laserscan_rays[4] / math.cos(laser_angle)
                remaining_front_laser = self.last_laserscan_rays[7] - front_laser
                laser_paral_dist = front_laser * math.sin(laser_angle)
                scalene_triang_known_angle = math.pi - laser_angle
                platform_side = math.sqrt(laser_paral_dist ** 2 + remaining_front_laser ** 2 - 2 * laser_paral_dist * remaining_front_laser * math.cos(scalene_triang_known_angle))
                delta_pitch = -math.asin(math.sin(scalene_triang_known_angle) * remaining_front_laser / platform_side)
            else:
                central_laser = self.last_laserscan_rays[1] * math.cos(laser_angle)
                laser_paral_dist = self.last_laserscan_rays[1] * math.sin(laser_angle)
                delta_pitch = -math.atan((self.last_laserscan_rays[4] - central_laser) / laser_paral_dist) 

        s += [delta_roll / self.max_delta_roll_pitch, 
              delta_pitch / self.max_delta_roll_pitch]

        delta_z = min(self.last_laserscan_rays)
        if not (self.last_laserscan_rays[4] == math.inf):
            delta_z = math.fabs(self.last_laserscan_rays[4] * math.cos(self.last_odom_rot[0]) * math.cos(self.last_odom_rot[1]))

        # if we are completely out of the platform, we use the world value, 
        # in the real world this would be a gps value
        if delta_z == math.inf:
            delta_z = self.last_odom_pos[2]

        s += [delta_z / self.max_delta_vertical]

        self.odom_updated = False
        self.laser_updated = False

        return np.array(s)

    def translate_action(self, action):
        a = Twist()
        a.linear.x = 0.0
        a.linear.y = 0.0
        a.linear.z = 0.0
        a.angular.x = 0.0
        a.angular.y = 0.0
        a.angular.z = 0.0

        if action == 0:
            a.linear.x = 1.0
        elif action == 1:
            a.linear.x = -1.0
        elif action == 2:
            a.linear.y = 1.0
        elif action == 3:
            a.linear.y = -1.0
        elif action == 4:
            a.linear.z = -1.0
        elif action == 5:
            a.linear.z = 1.0

        return a

    def calculate_reward(self):

        dist_eucl_pos = math.sqrt(
            (self.last_odom_pos[0] - self.platform_center[0]) ** 2 + 
            (self.last_odom_pos[1] - self.platform_center[1]) ** 2 + 
            (self.last_odom_pos[2] - self.platform_center[2]) ** 2)
        # normalization to (0, 1)
        dist_eucl_pos = dist_eucl_pos / math.sqrt(2 * (self.horizontal_bound ** 2) + self.vertical_bound ** 2)

        dist_eucl_vel = math.sqrt(
            (self.last_odom_vel_linear[0]) ** 2 + 
            (self.last_odom_vel_linear[1]) ** 2 + 
            (self.last_odom_vel_linear[2]) ** 2)
        # normalization to (0, 1)
        dist_eucl_vel = dist_eucl_vel / math.sqrt(2 * (self.max_horizontal_vel ** 2) + self.max_vertical_vel ** 2)

        dist_eucl_rot = math.sqrt(
            (self.last_odom_rot[0] - self.platform_rot[0]) ** 2 + 
            (self.last_odom_rot[1] - self.platform_rot[1]) ** 2)
        # normalization to (0, 1)
        dist_eucl_rot = dist_eucl_rot / (math.sqrt(2) * self.max_delta_roll_pitch)

        # combine the 3 rewards
        reward = - dist_eucl_pos - 0.2 * dist_eucl_vel - 0.1 * dist_eucl_rot

        if self.reset_flag:
            reward += -self.reward_penalty

        # multiply for a given value, needed only for more "easy to read" values
        reward *= self.reward_multiplier
        
        #self.node.get_logger().info("Rewards: tot %s, singoli: %s, %s, %s" % (reward, dist_eucl_pos, dist_eucl_vel, dist_eucl_rot))
        return reward

    # ---------------------------------------------------------------------------------------------
    # --- Callback functions ----------------------------------------------------------------------
    # ---------------------------------------------------------------------------------------------
    def execute_action(self, action):
        self.velocity_publisher.publish(action)

    def odom_callback(self, msg):
        self.skip_count_odom = (self.skip_count_odom + 1) % self.skip_msg
        if self.skip_count_odom != 0:
            return
        """
        Receive the odometry information containing the position and orientation
        of the robot in the global reference frame. 
        The position is x, y, z.
        The orientation is a x,y,z,w quaternion. 
        """
        roll, pitch, yaw = self._euler_from_quaternion(
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w)
        
        obs_state_vector_x_y_z = [msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z]
        obs_state_vector_p_r_y = [roll, pitch, yaw]
        
        self.last_odom_pos = obs_state_vector_x_y_z
        self.last_odom_rot = obs_state_vector_p_r_y
        self.last_odom_vel_linear = [msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.linear.z]

        self.odom_updated = True

        # position limits, the drone can't go too far away
        x_limit = obs_state_vector_x_y_z[0] > self.horizontal_bound or obs_state_vector_x_y_z[0] < -self.horizontal_bound
        y_limit = obs_state_vector_x_y_z[1] > self.horizontal_bound or obs_state_vector_x_y_z[1] < -self.horizontal_bound
        z_limit = obs_state_vector_x_y_z[2] > self.vertical_bound or obs_state_vector_x_y_z[2] < 0.0
        if x_limit or y_limit or z_limit:
            self.reset_flag = True

    def laser_callback(self, msg):
        self.skip_count_laser = (self.skip_count_laser + 1) % self.skip_msg
        if self.skip_count_laser != 0:
            return

        self.last_laserscan_rays = list(msg.ranges)        

        self.laser_updated = True

        '''
             A X    8 7 6
             |      5 4 3
        Y    |      2 1 0
        <----+      
        '''
        platform_touch = False
        for i in self.last_laserscan_rays:
            if i < 0.1:
                platform_touch = True

        # if we are touching the platform, but we are not in the same plane (roll-pitch) of the platform
        if platform_touch and (math.fabs(self.last_odom_rot[0] - self.platform_rot[0]) > 0.1 or math.fabs(self.last_odom_rot[1] - self.platform_rot[1]) > 0.1):
            self.reset_flag = True
        else:
            self.done_flag = self.done_flag or platform_touch

    def platform_callback(self, msg):
        """
        Receive the odometry information containing the position and orientation
        of the platform in the global reference frame. 
        The position is x, y, z.
        The orientation is a x,y,z,w quaternion. 
        """
        roll, pitch, _ = self._euler_from_quaternion(
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w)

        self.platform_rot = [roll, pitch]

    # ---------------------------------------------------------------------------------------------
    # --- Functions for gazebo node interactions --------------------------------------------------
    # ---------------------------------------------------------------------------------------------

    def spawn_entity(self, initial_pose):
        req = SpawnEntity.Request()
        req.name = self.quadcopter_name
        req.xml = self.model
        req.robot_namespace = self.quadcopter_namespace
        req.initial_pose = initial_pose
        self.node.get_logger().info('Calling service /spawn_entity')
        srv_call = self.client_spawn.call_async(req)
        while rclpy.ok():
            if srv_call.done():
                self.node.get_logger().info('Spawn status: Successfull')
                return
            rclpy.spin_once(self.node)
        self.node.get_logger().error('Call to /spawn_entity failed.')

    def delete_entity(self):
        req = DeleteEntity.Request()
        req.name = self.quadcopter_name
        self.node.get_logger().info('Calling service /delete_entity')
        srv_call = self.client_delete.call_async(req)
        while rclpy.ok():
            if srv_call.done():
                self.node.get_logger().info('Delete status: Successfull')
                return
            rclpy.spin_once(self.node)
        self.node.get_logger().error('Call to /delete_entity failed.')

    def unpause_physics(self):
        srv_call = self.client_unpause.call_async(Empty.Request())
        while rclpy.ok():
            if srv_call.done():
                self.node.get_logger().info('Unpause physics status: done')
                return
            rclpy.spin_once(self.node)
        self.node.get_logger().error('Call to /unpause_physics failed.')

    def pause_physics(self):
        srv_call = self.client_pause.call_async(Empty.Request())
        while rclpy.ok():
            if srv_call.done():
                self.node.get_logger().info('Pause physics status: done')
                return
            rclpy.spin_once(self.node)
        self.node.get_logger().error('Call to /pause_physics failed.')

    def reset_sim(self):
        srv_call = self.client_resetsim.call_async(Empty.Request())
        while rclpy.ok():
            if srv_call.done():
                self.node.get_logger().info('Reset simulation status: done')
                return
            rclpy.spin_once(self.node)
        self.node.get_logger().error('Call to /reset_sim failed.')

    def reset_world(self):
        srv_call = self.client_resetwrld.call_async(Empty.Request())
        while rclpy.ok():
            if srv_call.done():
                self.node.get_logger().info('Reset world status: done')
                return
            rclpy.spin_once(self.node)
        self.node.get_logger().error('Call to /reset_world failed.')

    # ---------------------------------------------------------------------------------------------
    # --- Utilities funcions ----------------------------------------------------------------------
    # ---------------------------------------------------------------------------------------------
    def are_values_updated(self):
        return self.odom_updated and self.laser_updated

    def print_odom_and_laser(self):
        self.node.get_logger().info("Odometry received:\n pos:\n %s\n rot:\n %s\n vel:\n %s\n laser:\n %s\n" % (self.last_odom_pos, self.last_odom_rot, self.last_odom_vel_linear, self.last_laserscan_rays))
    
    def _euler_from_quaternion(self, x, y, z, w):
        """
        Convert a quaternion into euler angles (roll, pitch, yaw)
        roll is rotation around x in radians (counterclockwise)
        pitch is rotation around y in radians (counterclockwise)
        yaw is rotation around z in radians (counterclockwise)
        """
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = math.atan2(t0, t1)

        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = math.asin(t2)

        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4)

        return roll_x, pitch_y, yaw_z # in radians