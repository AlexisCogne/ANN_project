import numpy as np
import torch
import torch.nn as nn
from torch import optim

# THIS A2C IS ADAPTED FROM https://github.com/hermesdt/reinforcement-learning/blob/master/a2c/cartpole_a2c_episodic.ipynb

class A2Cx(nn.Module):
    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        device: torch.device,
        critic_lr: float,
        actor_lr: float,
        n_envs: int,
    ):
        super().__init__()
        self.device = device
        self.n_envs = n_envs
        self.memory = Memory()

        #input is the state S(t), output is the probability distribution over the actions
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, n_actions),
            nn.Softmax()
        ).to(self.device)
        
        #input is the state S(t), output is the value of the state V(s)
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        ).to(self.device)
    
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)

    def forward(self, X):
        X = torch.Tensor(X).to(self.device)
        state_values = self.critic(X)
        action_pd = self.actor(X)
        return state_values, action_pd
    
    # train function
    def trainx(self,q_val,gamma):
        values = torch.stack(self.memory.values)
        q_vals = np.zeros((len(self.memory), 1))
        
        # target values are calculated backward
        # it's super important to handle correctly done states,
        # for those cases we want our to target to be equal to the reward only
        for i, (_, _, reward, done) in enumerate(self.memory.reversed()):
            q_val = reward + gamma*q_val*(1.0-done)
            q_vals[len(self.memory)-1 - i] = q_val # store values from the end to the beginning
            
        advantage = torch.Tensor(q_vals) - values
        
        critic_loss = advantage.pow(2).mean()
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        actor_loss = (-torch.stack(self.memory.log_probs)*advantage.detach()).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        


class Memory():
    def __init__(self):
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []

    def add(self, log_prob, value, reward, done):
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)
    
    def clear(self):
        self.log_probs.clear()
        self.values.clear()
        self.rewards.clear()
        self.dones.clear()  
    
    def _zip(self):
        return zip(self.log_probs,
                self.values,
                self.rewards,
                self.dones)
    
    def __iter__(self):
        for data in self._zip():
            return data
    
    def reversed(self):
        for data in list(self._zip())[::-1]:
            yield data
    
    def __len__(self):
        return len(self.rewards)