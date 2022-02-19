import sys

import numpy as np

import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup

from std_msgs.msg import Float32

from drone_worker.base import WorkerBase


class Worker(WorkerBase):
    """Worker node for generating and training experiences."""

    def __init__(
            self,
            #worker_id: int,
            name: str,
            output_file: str = 'test_model',
            policy_type: str = 'DQN') -> None:
        """Initialize Worker Node class."""
        super().__init__(name, policy_type)

        # Set the output file for final model.
        self.output_file = output_file

        # Set timer flags, futures, and callback groups.
        self._cb_group = ReentrantCallbackGroup()
        self._total_reward = 0
        self._update = 10
        
        # Set the number of episodes to zero.
        self.episode = 0

        # Create ROS publisher for episodes data.
        self._pub = self.create_publisher(
            Float32, 'episode_data', 10, callback_group=self._cb_group)

    def collect(self) -> int:
        """Collect a set of trajectories and store."""
        self.get_logger().info('Generating experiences...')
        # Decay epsilon if requred
        self._wp.decay_epsilon(self.episode, 0.001)

        # Generate experiences
        trajectory_length, total_reward = self.step()
        self.publish(total_reward)

        # Total reward section
        self._total_reward += total_reward
        if (self.episode + 1) % self._update == 0:
            self.get_logger().error(
                f'Expected Reward: {self._total_reward / self._update}')
            self._total_reward = 0

        return trajectory_length

    def compute(self, step: int) -> None:
        """Compute the gradient of the local network."""
        if self.episode >= self._wp.training_delay:
            
            if self.atype in ['DQN']:
                # Transfer network parameters if episode 0 or 100 * n.
                if self.episode % 100 == 0:
                    self._policy.transfer_parameters()

            self.get_logger().info('Computing gradients...')

            if self.atype in ['REINFORCE', 'A2C']:
                batch = self._db.sample_batch(step, 'all')
                self._policy.train(batch, step)

            else:
                batch = self._db.sample_batch(self._wp.batch_size)
                self._policy.train(batch, self._wp.batch_size)

        else:
            self.get_logger().warn(
                'Skipping computing gradients till episode ' +
                f'{self._wp.training_delay}!'
            )

        self.episode += 1

    def test(self, n_test_runs: int = 10) -> None:
        """Test the current network to check how well the networks trained."""
        steps: np.ndarray = np.zeros(n_test_runs)
        rewards: np.ndarray = np.zeros(n_test_runs)
        for t in range(n_test_runs):
            steps[t], rewards[t] = self.step(collect=False, testing=True)

        self.get_logger().warn('---------- TEST RUN RESULTS ----------')
        self.get_logger().warn(f'Average: {steps.mean()}')
        self.get_logger().warn(f'STD: {steps.std()}')
        self.get_logger().warn(f'Median: {np.median(steps)}')
        self.get_logger().warn(f'Average Reward: {rewards.mean()}')

    def publish(self, total_reward: float) -> None:
        """Publish the total reward for a experience trajectory."""
        msg = Float32()
        msg.data = total_reward
        self._pub.publish(msg)

    def upkeep(self) -> None:
        """Run policy dependent end-of-experiment upkeep on database, etc."""
        if self.atype in ['REINFORCE', 'A2C', 'PPO']:
            self._db.reset()

    def save_model(self) -> None:
        self._policy.save_model(self.output_file)


def main():
    """Start the Worker Node."""
    rclpy.init()

    output_file = sys.argv[1]
    policy_type = sys.argv[2]
    node = Worker('worker_node', output_file, policy_type)

    try:
        for i in range(node.episodes_number()):
            steps = node.collect()
            node.compute(steps)
            node.upkeep()

            if (i % 10 == 0):
                node.save_model()

        node.save_model()

        node.test(2)#100)

    except KeyboardInterrupt:
        pass

    # Destroy the node explicitly
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
