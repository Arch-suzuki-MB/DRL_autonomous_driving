import gym
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from env_v1 import ObstacleAvoidanceScenario
from torch.utils.tensorboard import SummaryWriter
import time 


#Defining some hyperparameters 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GAMMA=0.99
MAX_EPS=2000
TIME = time.strftime("%Y%m%d%H%M%S",time.localtime())
GRAD_CLIP_NORM = 5

class PolicyPi(nn.Module):
    def __init__(self, state_size,action_size,hidden_dim=[256,128,64]):
        super().__init__()

        self.hidden_1 = nn.Linear(state_size, hidden_dim[0])
        self.hidden_2 = nn.Linear(hidden_dim[0], hidden_dim[1])
        self.hidden_3 = nn.Linear(hidden_dim[1], hidden_dim[2])
        self.output = nn.Linear(hidden_dim[2], action_size)

    def forward(self, s):
        outs = F.relu(self.hidden_1(s))
        outs = F.relu(self.hidden_2(outs))
        outs = F.relu(self.hidden_3(outs))
        logits = self.output(outs)
        return logits



class Reinforce(nn.Module):
    def __init__(self):
        super().__init__() 
        self.env = ObstacleAvoidanceScenario()
        self.state_size = int(self.env.reset().size)
        self.action_size = self.env.action_space.n
        self.pi_policy =PolicyPi(self.state_size,self.action_size).to(device)
        self.gamma = GAMMA
        self.env = ObstacleAvoidanceScenario() 
        self.reward_records = []
        self.opt = torch.optim.AdamW(self.pi_policy.parameters(), lr=0.001)
        self.best_score = None
        self.writer = SummaryWriter(comment='-'+"Obstacle_avoidance" +TIME)

    # pick up action with above distribution policy_pi
    def pick_sample(self,s):
        with torch.no_grad():
            #   --> size : (1, 4)
            s_batch = np.expand_dims(s, axis=0)
            s_batch = torch.tensor(s_batch, dtype=torch.float).to(device)
            # Get logits from state
            #   --> size : (1, 2)
            logits = self.pi_policy(s_batch)
            #   --> size : (2)
            logits = logits.squeeze(dim=0)
            # From logits to probabilities
            probs = F.softmax(logits, dim=-1)
            # Pick up action's sample
            a = torch.multinomial(probs, num_samples=1)
            # Return
            return a.tolist()[0]

    def train(self):
        for i in range(MAX_EPS):
            #
            # Run episode till done
            #
            done = False
            states = []
            actions = []
            rewards = []
            s= self.env.reset()
            while not done:
                states.append(s.tolist())
                a = self.pick_sample(s)
                s, r, done,_ = self.env.step(a)
                actions.append(a)
                rewards.append(r)
                
            #
            # Get cumulative rewards
            #
            cum_rewards = np.zeros_like(rewards)
            reward_len = len(rewards)
            for j in reversed(range(reward_len)):
                cum_rewards[j] = rewards[j] + (cum_rewards[j+1]*self.gamma if j+1 < reward_len else 0)

            #saving the best model 
            self.reward_records.append(sum(rewards))
            if len(self.reward_records)>1 : 
                self.best_score = np.max(np.array(self.reward_records[:-1]))
            

                if self.best_score < self.reward_records[-1]:
                    torch.save(self.pi_policy.state_dict(), f"Reinforce_model_score{self.reward_records[-1]}.pth")

            self.writer.add_scalar("Avg_Rewards_100",np.mean(np.array(self.reward_records[-100:])),i)
            # Train (optimize parameters)
            #
            

            #
            # Train (optimize parameters)
            #
            states = torch.tensor(states, dtype=torch.float).to(device) #states now contain many s (batch of s)
            actions = torch.tensor(actions, dtype=torch.int64).to(device) #actions contain the batch of actions that have been exec in the env 
            cum_rewards = torch.tensor(cum_rewards, dtype=torch.float).to(device)
            self.opt.zero_grad()
            logits = self.pi_policy(states) 
            # Calculate negative log probability (-log P) as loss.
            # Cross-entropy loss is -log P in categorical distribution. (see above)
            log_probs = -F.cross_entropy(logits, actions, reduction="none") #-logits*log(PI(a/s))
            #print("log proba",log_probs.shape)
            #print("cum reward shape",cum_rewards.shape)
            loss = -log_probs * cum_rewards
            self.writer.add_scalar("Loss",loss.mean(),i)
            loss.sum().backward()
            #gradient cliping
            torch.nn.utils.clip_grad_norm_(self.pi_policy.parameters(),GRAD_CLIP_NORM)
            self.opt.step()
            # Output total rewards in episode (max 500)
            print("Run episode{} with rewards {}".format(i, sum(rewards)), end="\r")
            


        print("\nDone")
        self.env.close()

if __name__=="__main__":
    
    policy_pi = Reinforce().to(device)
    policy_pi.train()

