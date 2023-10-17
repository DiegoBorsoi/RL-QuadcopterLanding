import os
import sys
import yaml

import torch as th
import numpy as np

import rclpy
from geometry_msgs.msg import Twist

from drone_worker.gazebo_env_1D import DroneEnv1D
from drone_worker.gazebo_env_2D import DroneEnv2D
from drone_worker.gazebo_env import DroneEnv

from stable_baselines3 import PPO
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import safe_mean

from stable_baselines3.common.vec_env import DummyVecEnv

import dataclasses
from imitation.data import types
from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.algorithms import bc
from imitation.util import logger as imit_logger
from stable_baselines3.common.evaluation import evaluate_policy


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

    '''
    Callback for the manual input manager node
    '''
    def manual_input_callback(self, msg):
        #self.manual_node.get_logger().info("Input received: x: %s, y: %s, z: %s" % (msg.linear.x, msg.linear.y, msg.linear.z))
        self.last_manual_action = self.translate_action(msg.linear)

    '''
    Traslates the action from a Twist construct to the format used by the environment
    '''
    def translate_action(self, linear_action):
        action = -1
        x = linear_action.x
        y = linear_action.y
        z = linear_action.z

        if self.dimensions == 1:
            # actions: Down, Up, Wait
            action = 2
            if z == -1.0:
                action = 0
            elif z == 1.0:
                action = 1

        elif self.dimensions == 2:
            #actions: [left, wait, right], [down, wait, up]
            action = [1, 1]
            if y == 1.0:
                action[0] = 0
            elif y == -1.0:
                action[0] = 2

            if z == -1.0:
                action[1] = 0
            elif z == 1.0:
                action[1] = 2

        else:
            # Not managed
            pass

        return action

    '''
    Creates a collection of rollouts using a given network (loaded from load_path) and saves them on a given path
    '''
    def BC_create_rollout(self, ep_num, load_path, save_file_path):

        self.load(load_path)

        rng = np.random.default_rng(0)

        rollouts = rollout.rollout(
            self.model,
            DummyVecEnv([lambda: RolloutInfoWrapper(self.env)]),
            rollout.make_sample_until(min_timesteps=None, min_episodes=ep_num),
            rng=rng,
        )

        types.save(save_file_path, rollouts)

        return

    '''
    Function used for creating and saving rollouts using the manual input
    '''
    def BC_create_rollout_manual(self, ep_num, save_file_path):
        episodes = ep_num

        self.last_manual_action = self.translate_action(Twist().linear)

        self.manual_node = rclpy.create_node("manual_input_receiver")
        self.platform_subscriber = self.manual_node.create_subscription(Twist, '/manual_input', self.manual_input_callback, 1)

        rng = np.random.default_rng(0)
        
        venv = DummyVecEnv([lambda: RolloutInfoWrapper(self.env)])
        
        # Collect rollout tuples.
        rollouts = []
        # accumulator for incomplete trajectories
        trajectories_accum = rollout.TrajectoryAccumulator()
        obs = venv.reset()
        trajectories_accum.add_step(dict(obs=obs[0]), 0)

        while episodes > 0:
            rclpy.spin_once(self.manual_node)
            a = self.last_manual_action
            acts = np.array([a])
            obs, rews, dones, infos = venv.step(acts)

            new_trajs = trajectories_accum.add_steps_and_auto_finish(
                acts,
                obs,
                rews,
                dones,
                infos,
            )
            rollouts.extend(new_trajs)

            if dones[0]:
                episodes -= 1

        rng.shuffle(rollouts)

        rollouts = [rollout.unwrap_traj(traj) for traj in rollouts]
        rollouts = [dataclasses.replace(traj, infos=None) for traj in rollouts]

        types.save(save_file_path, rollouts)

        return

    '''
    Loads a collection of rollouts from a given path
    '''
    def BC_load_rollout(self, rollout_path):

        loaded_rollouts = types.load(rollout_path)
        
        return loaded_rollouts

    '''
    Trains a network for a given number of epochs using imitation learning on a given collection of roolouts.
    At the end of the training the neywork is saved on a given path
    '''
    def BC_train(self, rollouts, epochs, bc_trainer_path):
        
        transitions = rollout.flatten_trajectories(rollouts)

        rng = np.random.default_rng(0)
        
        bc_trainer = bc.BC(
            observation_space=self.env.observation_space,
            action_space=self.env.action_space,
            demonstrations=transitions,
            rng=rng,
            policy=self.model.policy,
            custom_logger=imit_logger.configure("BC_train_saves", ["stdout", "log", "tensorboard"])
        )

        bc_trainer.train(n_epochs=epochs)
        bc_trainer.save_policy(bc_trainer_path)

        return

    '''
    Tests a network for a given numebr of episodes, printing the results
    '''
    def BC_test(self, test_episodes_num, bc_trainer_path):
        bc_trained_policy = bc.reconstruct_policy(bc_trainer_path)

        reward, _ = evaluate_policy(bc_trained_policy, self.env, test_episodes_num, return_episode_rewards = True)
        print("BC trained agent reward:", reward)
        print("BC trained agent reward max:", max(reward))
        print("BC trained agent reward min:", min(reward))
        print("BC trained agent reward avg:", np.average(reward))

        c = 0
        for a in reward:
            if a > -1000:
                c += 1
        print("BC trained agent giusti (> -1000): " + str(c / len(reward) * 100) + "%")

        return

    '''
    Load a rollout, trains a network using the loaded experience and test the newly created network
    '''
    def BC_load_train_test(self, rollout_path, epochs, test_episodes_num, bc_trainer_path):

        roll = self.BC_load_rollout(rollout_path)
        self.BC_train(roll, epochs, bc_trainer_path)
        self.BC_test(test_episodes_num, bc_trainer_path)

        return

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

        ###
        # Examples of possible calls
        ###

        worker.learn()

        #worker.BC_create_rollout_manual(2, "saved-rollout/saved-rollout-manual.npz")
        
        #worker.BC_create_rollout(400, load_path, "saved-rollout/saved-rollout.npz")

        #worker.BC_load_train_test(
        #    "saved-rollout/saved-rollout_dim-2_plat-1_n-200_PPO-12_tot.npz", 400,
        #    "saved-rollout/bc_trainer_policy-PPO12_200-400.pt")

        #worker.BC_test(400, "saved-rollout/bc_trainer_policy-PPO12_200-400.pt") # media: 77.0 | 400 -> 77.0 79.5 79.75 76.75 75.25 77.5 78.75 77.25 74.25 77.25 73.25 75.75 78.75

        print("-----FINITO!!!!!------------------------------")

    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
