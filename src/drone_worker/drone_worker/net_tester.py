import sys

import numpy as np

import rclpy

from drone_worker.base import WorkerBase


class Tester(WorkerBase):
    """Tester node."""

    def __init__(
            self,
            #worker_id: int,
            name: str,
            input_folder: str = 'trained-models/save-1/',
            policy_type: str = 'DQN') -> None:
        """Initialize Worker Node class."""
        super().__init__(name, policy_type)

        # Set the input file for final model.
        self.input_folder = input_folder


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

    def load_model(self) -> None:
        self._policy.load_model(self.input_folder)


def main():
    """Start the Worker Node."""
    rclpy.init()

    input_folder = sys.argv[1]
    policy_type = sys.argv[2]
    try:
        n_test = int(sys.argv[3])
    except:
        n_test = 10

    node = Tester('tester_node', input_folder, policy_type)

    node.load_model()

    try:
        node.test(n_test)
    except KeyboardInterrupt:
        pass

    # Destroy the node explicitly
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
