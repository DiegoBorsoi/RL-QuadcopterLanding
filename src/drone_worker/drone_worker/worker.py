import os
import sys
import yaml

import torch as th

from drone_worker.gazebo_env_1D import DroneEnv1D
from drone_worker.gazebo_env import DroneEnv

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback


class SaveModelCallback(BaseCallback):
    """
    Callback for saving a model every ``save_skip`` episodes.

    :param save_skip: (int) number of timesteps (episodes) to skip before saving the model.
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """
    def __init__(self, save_skip: int, log_dir: str, verbose=1):
        super(SaveModelCallback, self).__init__(verbose)
        self.save_skip = save_skip
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'rl_model')

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.log_dir is not None:
            os.makedirs(self.log_dir, exist_ok=True)

    def _on_rollout_end(self) -> None:

        if self.num_timesteps % self.save_skip == 0:
            print(f"Saving new model to {self.save_path}.zip")
            self.model.save(self.save_path)

    def _on_step(self) -> bool:
        return True


class Worker():
    """Worker node for generating and training experiences."""

    def __init__(
            self,
            output_folder: str = 'saves-default/',
            policy_type: str = 'PPO',
            params = {"number": {"episode_max_steps": 300, 
                                 "episodes": 1000},
                      "hyperparameter": {"lr": 0.007,
                                         "gamma": 0.99,
                                         "epsilon": 1.0},
                      "hidden_layers": [64, 64]},
            dimensions: int = 3) -> None:

        # Set the output file for final model.
        self.output_folder = output_folder
        
        # Parameters passed using yaml config file
        # TODO: controllare/aggiungere parametri
        self.episode_max_steps = params["number"]['episode_max_steps']
        self.episodes_number = params["number"]['episodes']
        self.lr = params["hyperparameter"]['lr']
        self.gamma = params["hyperparameter"]['gamma']
        self.epsilon = params["hyperparameter"]['epsilon']
        self.hidden_layers = params['hidden_layers']

        print("Params: %s" % [self.episode_max_steps, self.episodes_number, self.lr, self.gamma, self.epsilon, self.hidden_layers])

        # Environment
        if dimensions == 1:
            self.env = DroneEnv1D(self.episode_max_steps)
        elif dimensions == 2:
            self.env = DroneEnv1D(self.episode_max_steps) # TODO: sistemare 2D
        else:
            self.env = DroneEnv(self.episode_max_steps)

        #print("Obs space: %s" % self.env.observation_space)
        #print("Action space: %s" % self.env.action_space)
        #print("Action sample: %s" % self.env.action_space.sample())

        self.save_callback = SaveModelCallback(save_skip=10 * self.episode_max_steps, log_dir=self.output_folder)
        # Model
        if policy_type == 'PPO':
            # arguments passed to the network
            policy_kwargs = dict(activation_fn=th.nn.Tanh,
                                 net_arch=[dict(pi=self.hidden_layers, vf=self.hidden_layers)])

            self.model = PPO("MlpPolicy", 
                             env = self.env,
                             learning_rate = self.lr, 
                             n_steps = self.episode_max_steps,
                             batch_size = 100,
                             verbose = 1,
                             tensorboard_log = "./tensorboard-test/",
                             policy_kwargs = policy_kwargs) # TODO: aggiungere parametri
        else:
            print("ERROR: Invalid policy: %s" % policy_type)

    def learn(self):
        # Learn for the given number of episodes
        self.model.learn(total_timesteps=self.episodes_number * self.episode_max_steps, 
                         callback=self.save_callback)


def main():
    output_folder = sys.argv[1]
    policy_type = sys.argv[2]
    params_file = sys.argv[3]
    dimensions = int(sys.argv[4])

    try:
        params = {}
        with open(params_file, 'r') as stream:
            try:
                params = yaml.safe_load(stream)
                params = params["worker_node"]
            except yaml.YAMLError as exc:
                print(exc)

        worker = Worker(output_folder, policy_type, params, dimensions)

        worker.learn()
        print("-----FINITO!!!!!------------------------------")

    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
