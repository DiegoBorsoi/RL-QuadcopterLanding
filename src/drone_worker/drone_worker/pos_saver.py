import os
import sys
import math

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry

class PosSaver(Node):

    def __init__(self, output_folder):
        super().__init__('pos_saver')

        self.output_file = output_folder + 'saved_pos'

        with open(self.output_file, 'w') as f:
            a = ""

        self.quadcopter_namespace = "demo"
        self.skip_msg = 1
        self.skip_count_odom = 0

        self.last_odom_pos = [0] * 3
        self.last_odom_rot = [0] * 3
        self.last_odom_vel_linear = [0] * 3

        self.odom_subscriber = self.create_subscription(Odometry,
                                                        '/' + self.quadcopter_namespace + '/odom',
                                                        self.odom_callback,
                                                        10)

        self.output_list_pos = []
        self.output_list_rot = []
        self.output_list_vel = []

        self.print_num = 20

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
        obs_state_vector_r_p_y = [roll, pitch, yaw]
        
        self.last_odom_pos = obs_state_vector_x_y_z
        self.last_odom_rot = obs_state_vector_r_p_y
        self.last_odom_vel_linear = [msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.linear.z]

        self.output_list_pos.append(self.last_odom_pos)
        self.output_list_rot.append(self.last_odom_rot)
        self.output_list_vel.append(self.last_odom_vel_linear)

        self.update_output()

    def update_output(self):
        if len(self.output_list_pos) < self.print_num:
            return

        with open(self.output_file, 'a') as f:
            f.write("".join([(str(i[0][0]) + "," + str(i[0][1]) + "," + str(i[0][2]) + "|" + str(i[1][0]) + "," + str(i[1][1]) + "," + str(i[1][2]) + "|" + str(i[2][0]) + "," + str(i[2][1]) + "," + str(i[2][2]) + '\n') for i in zip(self.output_list_pos, self.output_list_rot, self.output_list_vel)]))

        self.output_list_pos = []
        self.output_list_rot = []
        self.output_list_vel = []

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

def main():
    output_folder = sys.argv[1]

    rclpy.init()
    saver_node = PosSaver(output_folder)

    try:
        rclpy.spin(saver_node)

    except KeyboardInterrupt:
        pass

    saver_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
