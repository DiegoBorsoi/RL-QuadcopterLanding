import os
import sys
import yaml

import torch as th
import numpy as np

from drone_worker.gazebo_env_1D import DroneEnv1D
from drone_worker.gazebo_env_2D import DroneEnv2D
from drone_worker.gazebo_env import DroneEnv

from stable_baselines3 import PPO
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import safe_mean


class SaveModelCallback(BaseCallback):
    """
    Callback for saving a model every ``save_skip`` episodes.

    :param save_skip: (int) number of timesteps to skip before saving the model.
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """
    def __init__(self, save_rollout_skip: int, rollout_length: int, log_dir: str, verbose=1):
        super(SaveModelCallback, self).__init__(verbose)
        self.rollout_length = rollout_length
        self.save_steps_skip = save_rollout_skip * rollout_length
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'rl_model')

        self.best_mean_rew = -np.inf
        self.best_txt_file = os.path.join(log_dir, 'best_model_ep.txt')
        self.best_save_path = os.path.join(log_dir, 'rl_model-best')

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.log_dir is not None:
            os.makedirs(self.log_dir, exist_ok=True)

    def _on_rollout_end(self) -> None:

        # save model every self.save_skip rollouts
        if self.num_timesteps % self.save_steps_skip == 0:
            print(f"Saving new model to {self.save_path}.zip")
            self.model.save(self.save_path)

        # save best model
        rollout_mean_rew = safe_mean([ep_info["r"] for ep_info in self.locals["self"].ep_info_buffer])
        if rollout_mean_rew > self.best_mean_rew:
            # update mean reward and write it on a txt file
            self.best_mean_rew = rollout_mean_rew
            with open(self.best_txt_file, 'w') as f:
                f.write("ep: " + str(self.num_timesteps // self.rollout_length) + '\n')
                f.write("mean_rew: " + str(self.best_mean_rew) + '\n')
            # save the model
            self.model.save(self.best_save_path)

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
                      "hyperparameter": {"lr": 0.0003,
                                         "gamma": 0.99},
                      "hidden_layers": [64, 64]},
            dimensions: int = 3) -> None:

        self.policy_type = policy_type

        # Set the output file for final model.
        self.output_folder = output_folder
        
        # Parameters passed using yaml config file
        # TODO: controllare/aggiungere parametri
        self.episode_max_steps = params["number"]['episode_max_steps']
        self.episodes_number = params["number"]['episodes']
        self.lr = params["hyperparameter"]['lr']
        self.gamma = params["hyperparameter"]['gamma']
        self.hidden_layers = params['hidden_layers']

        print("Params: %s" % [self.episode_max_steps, self.episodes_number, self.lr, self.gamma, self.hidden_layers])

        self.dimensions = dimensions
        # Environment
        if self.dimensions == 1:
            self.env = DroneEnv1D(self.episode_max_steps)
        elif self.dimensions == 2:
            self.env = DroneEnv2D(self.episode_max_steps)
        else:
            self.env = DroneEnv(self.episode_max_steps)

        #print("Obs space: %s" % self.env.observation_space)
        #print("Action space: %s" % self.env.action_space)
        #print("Action sample: %s" % self.env.action_space.sample())

        self.save_callback = SaveModelCallback(save_rollout_skip=10, rollout_length=self.episode_max_steps, log_dir=self.output_folder)
        # Model
        if self.policy_type == 'PPO':
            # arguments passed to the network
            policy_kwargs = dict(activation_fn=th.nn.Tanh,
                                 net_arch=dict(pi=self.hidden_layers, vf=self.hidden_layers))

            self.model = PPO("MlpPolicy", 
                             env = self.env,
                             learning_rate = self.lr, 
                             n_steps = self.episode_max_steps,
                             batch_size = 100,
                             gamma = self.gamma,
                             verbose = 1,
                             tensorboard_log = "./tensorboard-test/",
                             #seed = 12345,
                             policy_kwargs = policy_kwargs) # TODO: aggiungere parametri
        elif self.policy_type == 'DQN':
            # arguments passed to the network
            policy_kwargs = dict(activation_fn=th.nn.ReLU,      # default used by stable-baselines3
                                 net_arch=self.hidden_layers)

            self.model = DQN("MlpPolicy", 
                             env = self.env,
                             #learning_rate = self.lr, 
                             train_freq = self.episode_max_steps,
                             batch_size = 100,
                             verbose = 1,
                             tensorboard_log = "./tensorboard-test/",
                             #seed = 12345,
                             policy_kwargs = policy_kwargs) # TODO: aggiungere parametri
        else:
            print("ERROR: Invalid policy: %s" % self.policy_type)

    def learn(self):
        # Learn for the given number of episodes
        self.model.learn(total_timesteps=self.episodes_number * self.episode_max_steps, 
                         callback=self.save_callback)

    def load(self, load_path: str = 'saves/'):

        load_file = os.path.join(load_path, 'rl_model')

        if self.policy_type == 'PPO':
            custom_objects = { 'learning_rate': self.lr }
            self.model = PPO.load(load_file, 
                                  env = self.env,
                                  custom_objects = custom_objects)
        elif self.policy_type == 'DQN':
            self.model = DQN.load(load_file, 
                                  env = self.env)
        else:
            print("ERROR: Invalid policy: %s" % self.policy_type)


def main():
    output_folder = sys.argv[1]
    policy_type = sys.argv[2]
    params_file = sys.argv[3]
    dimensions = int(sys.argv[4])
    load_net = sys.argv[5]
    load_path = sys.argv[6]

    try:
        params = {}
        with open(params_file, 'r') as stream:
            try:
                params = yaml.safe_load(stream)
                params = params["worker_node"]
            except yaml.YAMLError as exc:
                print(exc)

        worker = Worker(output_folder, policy_type, params, dimensions)

        if load_net == 'True':
            worker.load(load_path)

        worker.learn()
        print("-----FINITO!!!!!------------------------------")

    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
