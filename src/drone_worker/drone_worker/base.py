"""Base class for worker containing large initialization.

Written by: Zahi Kakish (zmk5)
"""
from typing import Tuple

import numpy as np

import os # Operating system library
import sys # Python runtime environment library
import math
import time
import random
import rclpy # ROS Client Library for Python
from rclpy.node import Node

from drone_worker.policy.dqn import WorkerPolicyDQN
from drone_worker.policy.ppo import WorkerPPO
from drone_worker.utils.database import LocalDatabase
from drone_worker.utils.parameters import Experience
from drone_worker.utils.parameters import WorkerParameters

# Package Management Library
from ament_index_python.packages import get_package_share_directory 

# Messages
from geometry_msgs.msg import Pose

# Twist is linear and angular velocity
from geometry_msgs.msg import Twist 

# LaserScan contains the value of the laser rays
from sensor_msgs.msg import LaserScan 
 
# Position, orientation, linear velocity, angular velocity
from nav_msgs.msg import Odometry

# Gazebo's services
from gazebo_msgs.srv import SpawnEntity 
from gazebo_msgs.srv import DeleteEntity 

from std_srvs.srv import Empty

class WorkerBase(Node):
    """Base class for worker containing large initialization."""

    def __init__(
            self,
            name: str,
            policy_type: str = 'DQN') -> None:
        """Initialize the WorkerBase Node class."""
        super().__init__(name)

        self.declare_parameters(
            namespace='',
            parameters=[
                ('use_gpu', False),
                ('number.states', 5),
                ('number.actions', 5),
                ('number.iterations', 1000),
                ('number.episodes', 5000),
                ('hyperparameter.alpha', 0.007),
                ('hyperparameter.beta', 0.1),
                ('hyperparameter.gamma', 0.99),
                ('hyperparameter.epsilon', 1.0),
                ('database.local_size', 10000),
                ('training_delay', 100),
                ('decay_rate', 500),
                ('batch_size', 16),
                ('hidden_layers', [16, 16]),
            ]
        )
        self._wp = WorkerParameters(
            self.get_parameter('number.states').value,
            self.get_parameter('number.actions').value,
            self.get_parameter('number.iterations').value,
            self.get_parameter('number.episodes').value,
            self.get_parameter('hyperparameter.alpha').value,
            self.get_parameter('hyperparameter.beta').value,
            self.get_parameter('hyperparameter.gamma').value,
            self.get_parameter('hyperparameter.epsilon').value,
            self.get_parameter('decay_rate').value,
            self.get_parameter('training_delay').value,
            self.get_parameter('batch_size').value
        )

        # Initialize local database for experience storage.
        self._db = LocalDatabase(
            self.get_parameter('database.local_size').value,
            self._wp.n_states,
            self._wp.n_iterations,
            np.float32
        )

        # Initialize policy for inference and training.
        if policy_type == 'DQN':
            self._policy = WorkerPolicyDQN(
                self._wp.n_states, self._wp.n_actions,
                self._wp.alpha, self._wp.gamma,
                self.get_parameter('hidden_layers').value,
                self.get_parameter('use_gpu').value
            )

        elif policy_type == 'PPO':
            self._policy = WorkerPPO(
                self._wp.n_states, self._wp.n_actions,
                self._wp.alpha, self._wp.gamma,
                self.get_parameter('hidden_layers').value,
                self.get_parameter('use_gpu').value
            )

        # --- Gazebo collector stuff --------------------------------------------------------------
        # Get path to the quadrotor
        sdf_file_path = os.path.join(get_package_share_directory("drone_worker"), "models", "quadrotor_v2", "model.sdf")

        try:
            self.model = open(sdf_file_path, 'r').read()
        except IOError as e:
            self.get_logger().error('Error reading file {}: {}'.format(sdf_file_path, e))
            return

        self.quadcopter_namespace = "demo"
        self.quadcopter_name = "Quadcopter"

        self.skip_msg = 100
        self.skip_count_odom = 0
        self.skip_count_laser = 0

        self.last_odom_pos = [0] * 3
        self.last_odom_rot = [0] * 3
        self.last_laserscan_rays = [0] * 9

        self.odom_updated = False
        self.laser_updated = False
        # -----------------------------------------------------------------------------------------

    @property
    def atype(self):
        """Return type of RL algorithm as string."""
        return self._policy.atype

    def episodes_number(self):
        return self._wp.n_episodes

    def step(self, collect: bool = True) -> Tuple[int, float]:
        """Generate experiences using the mean-field model for a policy."""
        
        #self._policy.act(state, epsilon)
        '''
        total_reward = 0
        
        old_state = np.random.rand(self._wp.n_states)
        
        for i in range(1, self._wp.n_iterations + 1):
            new_exp = Experience(self._wp.n_states)
            
            next_state = np.random.rand(self._wp.n_states)
            new_exp.new_experience(
                i,
                old_state,
                np.random.randint(0, self._wp.n_actions, 1).item(),
                np.random.rand(1).item(),
                next_state,
                np.random.rand(1).item(),
                i % (self._wp.n_iterations // 20) == 0
            )
            old_state = next_state
            total_reward = total_reward + new_exp.reward
            
            if collect:
                self._db.save_experience(new_exp, new_exp.done == 1)
        
        return (self._wp.n_iterations, total_reward)
        '''

        '''
        success = self._spawn_entity(initial_pose)
        if not success:
            self.get_logger().error('Spawn service failed. Exiting.')
            return (0, 0)
        '''
        # Create a subscriber
        # This node subscribes to messages of type
        # nav_msgs/Odometry (i.e. position and orientation of the robot)
        self.odom_subscriber = self.create_subscription(Odometry,
                                                        '/' + self.quadcopter_namespace + '/odom',
                                                        self.odom_callback,
                                                        10)
        # Create a subscriber
        # This node subscribes to messages of type
        # sensor_msgs/LaserScan (i.e. values of laser rays)
        self.laser_subscriber = self.create_subscription(LaserScan,
                                                        '/' + self.quadcopter_namespace + '/laser',
                                                        self.laser_callback,
                                                        10)
        
        # Create a publisher
        # This node publishes the desired linear and angular velocity of the robot (in the
        # robot chassis coordinate frame) to the /demo/cmd_vel topic. Using the diff_drive
        # plugin enables the robot model to read this /demo/cmd_vel topic and execute
        # the motion accordingly.
        self.velocity_publisher = self.create_publisher(Twist,
                                                        '/' + self.quadcopter_namespace + '/cmd_vel',
                                                        10)

        initial_pose = Pose()

        total_reward = 0.0

        run_steps = 0
        run_max_steps = 300

        self.reset = True
        self.done = False
        for i in range(0, self._wp.n_iterations + 1):
            
            if self.reset or self.done:
                initial_pose.position.x = random.uniform(-1, 1)
                initial_pose.position.y = random.uniform(-1, 1)
                initial_pose.position.z = random.uniform(2, 3)
                self._spawn_entity(initial_pose)
                self.unpause_physics()
                self.reset = False
                self.done = False
                old_state = np.array([])
                old_action = -1

            # wait for callback new values
            while not self.are_values_updated():
                rclpy.spin_once(self)

            # get the new experience
            state = self.get_state()
            action = self._policy.act(np.expand_dims(state, axis=0))
            reward = self.calculate_reward()
            total_reward += reward
            #self.get_logger().info("State:\n %s\naction:\n %s\n" % (state, action))

            if collect and not (old_state.size == 0 or old_action == -1):
                new_exp = Experience(self._wp.n_states)
                new_exp.new_experience(
                    i,
                    old_state,
                    old_action,
                    reward,
                    state,
                    self._policy.act(np.expand_dims(state, axis=0)),
                    self.done)
                self._db.save_experience(new_exp, self.done)
            self.get_logger().info("Iter: %s, reward: %s" % (i, reward))

            # Each run can last at max run_max_steps iterations
            if run_steps >= run_max_steps:
                self.reset = True
                run_steps = 0
            else:
                run_steps += 1

            if not (self.done or self.reset):
                old_state = state
                old_action = action

                self.execute_action(self.translate_action(action))
            else:
                # if done reset world and model
                self.pause_physics()
                self.delete_entity()
                time.sleep(0.2)
                self.reset_world()
                self.reset_sim()

        if not (self.done or self.reset):
            self.pause_physics()
            self.delete_entity()
            time.sleep(0.2)
            self.reset_world()
            self.reset_sim()

        return (self._wp.n_iterations, total_reward)

    # ---------------------------------------------------------------------------------------------
    # --- RL Functions ----------------------------------------------------------------------------
    # ---------------------------------------------------------------------------------------------
    def get_state(self):
        s = [self.last_odom_rot[0], self.last_odom_rot[1]] + self.last_odom_vel_linear

        out_front = self.last_laserscan_rays[8] == math.inf
        out_back = self.last_laserscan_rays[2] == math.inf
        out_left = self.last_laserscan_rays[6] == math.inf
        out_right = self.last_laserscan_rays[4] == math.inf

        s += [int(out_front), int(out_back), int(out_left), int(out_right)]

        laser_angle = math.pi / 12

        delta_roll = 0
        if s[0] > 0: # inclined in the leftward direction
            if out_left + (self.last_laserscan_rays[5] == math.inf) + out_right > 1:
                delta_roll = math.pi / 2
            elif out_left:
                right_laser = self.last_laserscan_rays[5] / math.cos(laser_angle)
                remaining_right_laser = self.last_laserscan_rays[4] - right_laser
                laser_paral_dist = right_laser * math.sin(laser_angle)
                scalene_triang_known_angle = math.pi - laser_angle
                platform_side = math.sqrt(laser_paral_dist ** 2 + remaining_right_laser ** 2 - 2 * laser_paral_dist * remaining_right_laser * math.cos(scalene_triang_known_angle))
                delta_roll = math.asin(math.sin(scalene_triang_known_angle) * remaining_right_laser / platform_side)
            else:
                central_laser = self.last_laserscan_rays[6] * math.cos(laser_angle)
                laser_paral_dist = self.last_laserscan_rays[6] * math.sin(laser_angle)
                delta_roll = math.atan((self.last_laserscan_rays[5] - central_laser) / laser_paral_dist) 
        elif s[0] < 0: # inclined in the rightward direction
            if out_left + (self.last_laserscan_rays[5] == math.inf) + out_right > 1:
                delta_roll = -math.pi / 2
            elif out_right:
                left_laser = self.last_laserscan_rays[5] / math.cos(laser_angle)
                remaining_left_laser = self.last_laserscan_rays[6] - front_laser
                laser_paral_dist = left_laser * math.sin(laser_angle)
                scalene_triang_known_angle = math.pi - laser_angle
                platform_side = math.sqrt(laser_paral_dist ** 2 + remaining_left_laser ** 2 - 2 * laser_paral_dist * remaining_left_laser * math.cos(scalene_triang_known_angle))
                delta_roll = -math.asin(math.sin(scalene_triang_known_angle) * remaining_left_laser / platform_side)
            else:
                central_laser = self.last_laserscan_rays[4] * math.cos(laser_angle)
                laser_paral_dist = self.last_laserscan_rays[4] * math.sin(laser_angle)
                delta_roll = -math.atan((self.last_laserscan_rays[5] - central_laser) / laser_paral_dist) 

        delta_pitch = 0
        if s[1] > 0: # inclined in the forward direction
            if out_front + (self.last_laserscan_rays[5] == math.inf) + out_back > 1:
                delta_pitch = math.pi / 2
            elif out_front:
                back_laser = self.last_laserscan_rays[5] / math.cos(laser_angle)
                remaining_back_laser = self.last_laserscan_rays[2] - back_laser
                laser_paral_dist = back_laser * math.sin(laser_angle)
                scalene_triang_known_angle = math.pi - laser_angle
                platform_side = math.sqrt(laser_paral_dist ** 2 + remaining_back_laser ** 2 - 2 * laser_paral_dist * remaining_back_laser * math.cos(scalene_triang_known_angle))
                delta_pitch = math.asin(math.sin(scalene_triang_known_angle) * remaining_back_laser / platform_side)
            else:
                central_laser = self.last_laserscan_rays[8] * math.cos(laser_angle)
                laser_paral_dist = self.last_laserscan_rays[8] * math.sin(laser_angle)
                delta_pitch = math.atan((self.last_laserscan_rays[5] - central_laser) / laser_paral_dist) 
        elif s[1] < 0: # inclined in the backward direction
            if out_front + (self.last_laserscan_rays[5] == math.inf) + out_back > 1:
                delta_pitch = -math.pi / 2
            elif out_back:
                front_laser = self.last_laserscan_rays[5] / math.cos(laser_angle)
                remaining_front_laser = self.last_laserscan_rays[8] - front_laser
                laser_paral_dist = front_laser * math.sin(laser_angle)
                scalene_triang_known_angle = math.pi - laser_angle
                platform_side = math.sqrt(laser_paral_dist ** 2 + remaining_front_laser ** 2 - 2 * laser_paral_dist * remaining_front_laser * math.cos(scalene_triang_known_angle))
                delta_pitch = -math.asin(math.sin(scalene_triang_known_angle) * remaining_front_laser / platform_side)
            else:
                central_laser = self.last_laserscan_rays[2] * math.cos(laser_angle)
                laser_paral_dist = self.last_laserscan_rays[2] * math.sin(laser_angle)
                delta_pitch = -math.atan((self.last_laserscan_rays[5] - central_laser) / laser_paral_dist) 

        s += [delta_pitch, delta_roll]

        delta_z = min(self.last_laserscan_rays)
        if not (self.last_laserscan_rays[5] == math.inf):
            delta_z = math.fabs(self.last_laserscan_rays[5] * math.cos(s[0]) * math.cos(s[1]))

        # if we are completely out of the platform, we use the world value, 
        # in the real world this would be a gps value
        if delta_z == math.inf:
            delta_z = self.last_odom_pos[2]

        s += [delta_z]

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
        platform_center = [0.0, 0.0, 0.2 + 0.1]

        dist_eucl_pos = math.fabs(
            (self.last_odom_pos[0] - platform_center[0]) ** 2 + 
            (self.last_odom_pos[1] - platform_center[1]) ** 2 + 
            (self.last_odom_pos[2] - platform_center[2]) ** 2)

        dist_eucl_vel = math.fabs(
            (self.last_odom_vel_linear[0]) ** 2 + 
            (self.last_odom_vel_linear[1]) ** 2 + 
            (self.last_odom_vel_linear[2]) ** 2)

        platform_rot = [0.0, 0.0]
        dist_eucl_rot = math.fabs(
            (self.last_odom_rot[0] - platform_rot[0]) ** 2 + 
            (self.last_odom_rot[1] - platform_rot[1]) ** 2)

        reward = - dist_eucl_pos - 0.2 * dist_eucl_vel - 0.1 * dist_eucl_rot

        if self.reset:
            reward += -500.0
        
        return reward

    # ---------------------------------------------------------------------------------------------
    # --- Callback functions ----------------------------------------------------------------------
    # ---------------------------------------------------------------------------------------------
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

        horizontal_bound = 5.0
        # position limits, the drone can't go too far away
        x_limit = obs_state_vector_x_y_z[0] > horizontal_bound or obs_state_vector_x_y_z[0] < -horizontal_bound
        y_limit = obs_state_vector_x_y_z[1] > horizontal_bound or obs_state_vector_x_y_z[1] < -horizontal_bound
        z_limit = obs_state_vector_x_y_z[2] > 4.0 or obs_state_vector_x_y_z[2] < 0.0
        if x_limit or y_limit or z_limit:
            self.reset = True

    def laser_callback(self, msg):
        self.skip_count_laser = (self.skip_count_laser + 1) % self.skip_msg
        if self.skip_count_laser != 0:
            return

        rays = msg.ranges

        self.last_laserscan_rays = list(rays)        

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

        platform_rot = [0.0, 0.0]
        # if we are touching the platform, but we are not in the same plane (roll-pitch) of the platform
        if platform_touch and (math.fabs(self.last_odom_rot[0] - platform_rot[0]) > 0.1 or math.fabs(self.last_odom_rot[1] - platform_rot[1]) > 0.1):
            self.reset = True
        else:
            self.done = self.done or platform_touch

    # ---------------------------------------------------------------------------------------------
    # --- Functions for gazebo node interactions --------------------------------------------------
    # ---------------------------------------------------------------------------------------------
    def execute_action(self, action):
        self.velocity_publisher.publish(action)

    def _spawn_entity(self, initial_pose, timeout=5.0):

        client = self.create_client(SpawnEntity, '/spawn_entity')
        if client.wait_for_service(timeout_sec=timeout):
            req = SpawnEntity.Request()
            req.name = self.quadcopter_name
            req.xml = self.model
            req.robot_namespace = self.quadcopter_namespace
            req.initial_pose = initial_pose
            self.get_logger().info('Calling service /spawn_entity')
            srv_call = client.call_async(req)
            while rclpy.ok():
                if srv_call.done():
                    self.get_logger().info('Spawn status: %s' % srv_call.result().status_message)
                    break
                rclpy.spin_once(self)
            return srv_call.result().success
        self.get_logger().error('Service /spawn_entity unavailable. Was Gazebo started with GazeboRosFactory?')
        return False

    def delete_entity(self, timeout=5.0):
        client = self.create_client(DeleteEntity, '/delete_entity')
        if client.wait_for_service(timeout_sec=timeout):
            req = DeleteEntity.Request()
            req.name = self.quadcopter_name
            self.get_logger().info('Calling service /delete_entity')
            srv_call = client.call_async(req)
            while rclpy.ok():
                if srv_call.done():
                    self.get_logger().info('Spawn status: %s' % srv_call.result().status_message)
                    break
                rclpy.spin_once(self)
            return srv_call.result().success
        self.get_logger().error('Service /delete_entity unavailable. Was Gazebo started with GazeboRosFactory?')
        return False

    def unpause_physics(self, timeout=5.0):
        client = self.create_client(Empty, '/unpause_physics')
        if client.wait_for_service(timeout_sec=timeout):
            srv_call = client.call_async(Empty.Request())
            while rclpy.ok():
                if srv_call.done():
                    self.get_logger().info('Unpause physics status: done')
                    break
                rclpy.spin_once(self)
            return True
        self.get_logger().error('Service /unpause_physics unavailable. Was Gazebo started right?')
        return False

    def pause_physics(self, timeout=5.0):
        client = self.create_client(Empty, '/pause_physics')
        if client.wait_for_service(timeout_sec=timeout):
            srv_call = client.call_async(Empty.Request())
            while rclpy.ok():
                if srv_call.done():
                    self.get_logger().info('Pause physics status: done')
                    break
                rclpy.spin_once(self)
            return True
        self.get_logger().error('Service /pause_physics unavailable. Was Gazebo started right?')
        return False

    def reset_sim(self, timeout=5.0):
        client = self.create_client(Empty, '/reset_simulation')
        if client.wait_for_service(timeout_sec=timeout):
            srv_call = client.call_async(Empty.Request())
            while rclpy.ok():
                if srv_call.done():
                    self.get_logger().info('Reset simulation status: done')
                    break
                rclpy.spin_once(self)
            return True
        self.get_logger().error('Service /reset_simulation unavailable. Was Gazebo started right?')
        return False

    def reset_world(self, timeout=5.0):
        client = self.create_client(Empty, '/reset_world')
        if client.wait_for_service(timeout_sec=timeout):
            srv_call = client.call_async(Empty.Request())
            while rclpy.ok():
                if srv_call.done():
                    self.get_logger().info('Reset world status: done')
                    break
                rclpy.spin_once(self)
            return True
        self.get_logger().error('Service /reset_world unavailable. Was Gazebo started right?')
        return False

    # ---------------------------------------------------------------------------------------------
    # --- Utilities funcions ----------------------------------------------------------------------
    # ---------------------------------------------------------------------------------------------
    def are_values_updated(self):
        return self.odom_updated and self.laser_updated

    def print_odom_and_laser(self):

        self.get_logger().info("Odometry received:\n pos:\n %s\n rot:\n %s\n vel:\n %s\n laser:\n %s\n" % (self.last_odom_pos, self.last_odom_rot, self.last_odom_vel_linear, self.last_laserscan_rays))

        self.odom_updated = False
        self.laser_updated = False
    
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
