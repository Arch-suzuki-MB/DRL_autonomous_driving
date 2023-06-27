
import os
import sys
from os.path import dirname, abspath
sys.path.append(dirname(dirname(abspath(__file__))))


import gym
import time
from Dueling_DDQN import Dueling_DDQN
from Trainer import Trainer
from utilities.data_structures.Config import Config
from DDQN import DDQN
from DDQN_With_Prioritised_Experience_Replay import DDQN_With_Prioritised_Experience_Replay
from DQN import DQN
from DQN_With_Fixed_Q_Targets import DQN_With_Fixed_Q_Targets


from A2C import A2C
from SAC_Discrete import SAC_Discrete
from A3C import A3C

## envs import ##
from env_v1 import ObstacleAvoidanceScenario
import env_v1 
import env_v1_dynamic 
env_title = "ObstacleAvoidance-v0"

env_v1.register()
#env_v1_dynamic.register()

config = Config()
config.env_title = env_title
config.seed = 1
config.environment = gym.make(env_title)
config.num_episodes_to_run = 2000
config.show_solution_score = False
config.visualise_individual_results = True
config.visualise_overall_agent_results = True
config.standard_deviation_results = 1.0
config.runs_per_agent = 1
config.use_GPU = True
config.overwrite_existing_results_file = False
config.randomise_random_seed = True
config.save_model = True
config.log_loss = False
config.log_base = time.strftime("%Y%m%d%H%M%S", time.localtime())
config.save_model_freq = 300    ## save model per 300 episodes

config.retrain = True
config.resume = True
config.resume_path = r"C:\Users\safae\OneDrive\Bureau\my project\Models\ObstacleAvoidance-v0\DDQN with Prioritised Replay\20230418135036\rolling_score_32.3000.model"
config.backbone_pretrain = False

config.force_explore_mode = True
config.force_explore_stare_e = 0.4 ## when the std of rolling score in last 10 window is smaller than this val, start explore mode
config.force_explore_rate = 0.95 ## only when the current score bigger than 0.8*max(rolling score[-10:]), forece expolre

## data and graphs save dir ##
data_results_root = os.path.join(os.path.dirname(__file__)+"/data_and_graphs/carla_obstacle_avoidance", config.log_base)
while os.path.exists(data_results_root):
    data_results_root += '_'
os.makedirs(data_results_root)
config.file_to_save_data_results = os.path.join(data_results_root, "data.pkl")
config.file_to_save_results_graph = os.path.join(data_results_root, "data.png")


config.hyperparameters = {
    "DQN_Agents": {
        "learning_rate": 1e-1,
        "batch_size": 32,
        "buffer_size": 20000,
        "epsilon": 1.0,
        "epsilon_decay_rate_denominator": 1.,
        "discount_rate": 0.9,
        "tau": 0.01,
        "alpha_prioritised_replay": 0.6,
        "beta_prioritised_replay": 0.1,
        "incremental_td_error": 1e-8,
        "update_every_n_steps": 1,
        "linear_hidden_units": [24, 48, 24],
        "final_layer_activation": "None",
        "batch_norm": False,
        "gradient_clipping_norm": 0.1,
        "learning_iterations": 1,
        "clip_rewards": False
    },
    
    "Actor_Critic_Agents":  {

        "learning_rate": 0.005,
        "linear_hidden_units": [20, 10],
        "final_layer_activation": ["SOFTMAX", None],
        "gradient_clipping_norm": 5.0,
        "discount_rate": 0.99,
        "epsilon_decay_rate_denominator": 1.0,
        "normalise_rewards": True,
        "exploration_worker_difference": 2.0,
        "clip_rewards": False,

        "Actor": {
            "learning_rate": 0.0003,
            "linear_hidden_units": [64, 64],
            "final_layer_activation": "Softmax",
            "batch_norm": False,
            "tau": 0.005,
            "gradient_clipping_norm": 5,
            "initialiser": "Xavier"
        },

        "Critic": {
            "learning_rate": 0.0003,
            "linear_hidden_units": [64, 64],
            "final_layer_activation": None,
            "batch_norm": False,
            "buffer_size": 1000000,
            "tau": 0.005,
            "gradient_clipping_norm": 5,
            "initialiser": "Xavier"
        },

        "min_steps_before_learning": 400,
        "batch_size": 256,
        "discount_rate": 0.99,
        "mu": 0.0, #for O-H noise
        "theta": 0.15, #for O-H noise
        "sigma": 0.25, #for O-H noise
        "action_noise_std": 0.2,  # for TD3
        "action_noise_clipping_range": 0.5,  # for TD3
        "update_every_n_steps": 1,
        "learning_updates_per_learning_session": 1,
        "automatically_tune_entropy_hyperparameter": True,
        "entropy_term_weight": None,
        "add_extra_noise": False,
        "do_evaluation_iterations": True
    }
    
}

if __name__ == "__main__":
    # AGENTS = [SAC_Discrete, DDQN, Dueling_DDQN, DQN, DQN_With_Fixed_Q_Targets,
    #           DDQN_With_Prioritised_Experience_Replay, A2C, PPO, A3C ]
    AGENTS = [A3C]
    trainer = Trainer(config, AGENTS)
    trainer.run_games_for_agents()
    pass