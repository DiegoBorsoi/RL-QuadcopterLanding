import sys # Python runtime environment library
import rclpy # ROS Client Library for Python
from rclpy.node import Node

from std_msgs.msg import Float32

class Saver(Node):

	def __init__(self, output_folder):
		"""
		Class constructor to set up the node
		"""
		##################### ROS SETUP ####################################################
		# Initiate the Node class's constructor and give it a name
		super().__init__('saver')

		self._sub = self.create_subscription(Float32, 'episode_data', self.data_callback, 10)

		self.output_file = output_folder + 'saved_reward'

		with open(self.output_file, 'w') as f:
			a = ""

		self.last_data = []

	def data_callback(self, msg):
		self.last_data.append(msg.data)

		if (len(self.last_data) >= 1):
			with open(self.output_file, 'a') as f:
				f.write(" ".join([str(i) for i in self.last_data]) + '\n')

			self.last_data = []


def main():
	rclpy.init()

	output_folder = sys.argv[1]
	save_node = Saver(output_folder)

	rclpy.spin(save_node)

	save_node.destroy_node()
	rclpy.shutdown()
	sys.exit(exit_code)


if __name__ == '__main__':
	main()
