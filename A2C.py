import gym
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F 
from env_v1 import ObstacleAvoidanceScenario
from torch.utils.tensorboard import SummaryWriter
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GAMMA=0.99
LR=0.001
MAX_EPS = 2000
GRAD_CLIP_NORM = 5
TIME = time.strftime("%Y%m%d%H%M%S",time.localtime())


class ActorNet(nn.Module):
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

class ValueNet(nn.Module):
    def __init__(self,state_size, hidden_dim=[256,128,64]):
        super().__init__()

        self.hidden_1 = nn.Linear(state_size, hidden_dim[0])
        self.hidden_2 = nn.Linear(hidden_dim[0], hidden_dim[1])
        self.hidden_3 = nn.Linear(hidden_dim[1], hidden_dim[2])
        self.output = nn.Linear(hidden_dim[2], 1)

    def forward(self, s):
        outs = F.relu(self.hidden_1(s))
        outs = F.relu(self.hidden_2(outs))
        outs = F.relu(self.hidden_3(outs))
        value = self.output(outs)
        return value


class A2C(nn.Module):
    def __init__(self):
        super().__init__() 
        self.env = ObstacleAvoidanceScenario()
        self.state_size = int(self.env.reset().size)
        self.action_size = self.env.action_space.n
        self.actor_func = ActorNet(self.state_size,self.action_size).to(device)
        self.value_func = ValueNet(self.state_size).to(device)
        self.gamma = GAMMA
        self.env = ObstacleAvoidanceScenario() #gym.make("CartPole-v1")
        self.reward_records = []
        self.opt1 = torch.optim.AdamW(self.value_func.parameters(), lr=LR)
        self.opt2 = torch.optim.AdamW(self.actor_func.parameters(), lr=LR)
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
            logits = self.actor_func(s_batch)
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
                    torch.save(self.actor_func.state_dict(), f"a3c_value_model_score{self.reward_records[-1]}.pth")
                    torch.save(self.value_func.state_dict(), f"a3c_policy_model_score{self.reward_records[-1]}.pth")

            self.writer.add_scalar("Avg_Rewards_100",np.mean(np.array(self.reward_records[-100:])),i)
            # Train (optimize parameters)
            #
            

            # Optimize value loss (Critic)
            self.opt1.zero_grad()
            states = torch.tensor(states, dtype=torch.float).to(device)
            cum_rewards = torch.tensor(cum_rewards, dtype=torch.float).to(device)
            values = self.value_func(states)
            print(values.shape)
            values = values.squeeze(dim=1)
            print(values.shape)
            vf_loss = F.mse_loss(values,cum_rewards,reduction="none")
            self.writer.add_scalar("Critic_loss",vf_loss.mean(),i)
            vf_loss.mean().backward()
            #gradient cliping
            torch.nn.utils.clip_grad_norm_(self.value_func.parameters(),GRAD_CLIP_NORM)
            self.opt1.step()

            # Optimize policy loss (Actor)
            with torch.no_grad():
                values = self.value_func(states)
            self.opt2.zero_grad()
            actions = torch.tensor(actions, dtype=torch.int64).to(device)
            advantages = cum_rewards - values
            logits = self.actor_func(states)
            log_probs = -F.cross_entropy(logits, actions, reduction="none")
            pi_loss = -log_probs * advantages
            self.writer.add_scalar("Actor_loss",pi_loss.mean(),i)
            pi_loss.mean().backward()
            torch.nn.utils.clip_grad_norm_(self.actor_func.parameters(),GRAD_CLIP_NORM)
            self.opt2.step()

            # Output total rewards in episode (max 500)
            print("Run episode{} with rewards {}".format(i, sum(rewards)), end="\r")
            


        print("\nDone")
        self.env.close()

if __name__=="__main__":
    a2c= A2C()
    a2c.train()
