import os
import sys
import yaml

from drone_worker.gazebo_env import DroneEnv

from stable_baselines3 import PPO

from stable_baselines3.common.evaluation import evaluate_policy


class Tester():
    """Worker node for generating and training experiences."""

    def __init__(
            self,
            input_folder: str = 'saves-default/',
            policy_type: str = 'PPO',
            params = {"number": {"episode_max_steps": 300, 
                                 "episodes": 1000},
                      "hyperparameter": {"lr": 0.007,
                                         "gamma": 0.99,
                                         "epsilon": 1.0},
                      "hidden_layers": [64, 64]}) -> None:

        # Set the output file for final model.
        self.input_folder = input_folder
        
        # Parameters passed using yaml config file
        # TODO: controllare/aggiungere parametri
        self.episode_max_steps = params["number"]["episode_max_steps"]
        self.episodes_number = params["number"]["episodes"]
        self.lr = params["hyperparameter"]["lr"]
        self.gamma = params["hyperparameter"]["gamma"]
        self.epsilon = params["hyperparameter"]["epsilon"]
        self.hidden_layers = params["hidden_layers"]

        print("Params: %s" % [self.episode_max_steps, self.episodes_number, self.lr, self.gamma, self.epsilon, self.hidden_layers])

        # Environment
        self.env = DroneEnv(self.episode_max_steps)

        # Model
        if policy_type == 'PPO':
            self.model = PPO.load(os.path.join(self.input_folder, 'rl_model'))
        else:
            print("ERROR: Invalid policy: %s" % policy_type)

    def test(self, n_test = 10):
        ris = evaluate_policy(model = self.model, 
                              env = self.env,
                              n_eval_episodes = n_test,
                              deterministic = True,
                              render = False, # render is always false, because is taken care of by gzclient
                              return_episode_rewards = True) # if True -> return the rewards of all the episodes

        print('---------- TEST RUN RESULTS ----------')
        print(f'Ris: {ris}')



def main():
    input_folder = sys.argv[1]
    policy_type = sys.argv[2]
    params_file = sys.argv[3]
    try:
        n_test = int(sys.argv[4])
    except:
        n_test = 10

    try:
        params = {}
        with open(params_file, 'r') as stream:
            try:
                params = yaml.safe_load(stream)
                params = params["tester_node"]
            except yaml.YAMLError as exc:
                print(exc)

        tester = Tester(input_folder, policy_type, params)
        tester.test(n_test)

        print("-----FINITO!!!!!------------------------------")
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
