import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from A2C_discrete_kworkers import A2C
from tqdm import tqdm
import os


# AGENT HYPERPARAMETERS
gamma = 0.99  # discount factor
ent_coef = 0.01  # coefficient for the entropy bonus (to encourage exploration)
actor_lr = 1e-5
critic_lr = 1e-3
stochasticity_bool = True
stochastic_reward_probability = 0.9
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# CREATE ENVIRONMENT
env = gym.make("CartPole-v1")
obs_shape = env.observation_space.shape[0]
action_shape = env.action_space.n

# LOAD WEIGHTS
actor_weights_path = "weights/actor_weights_stoch.h5"
critic_weights_path = "weights/critic_weights_stoch.h5"

agent = A2C(obs_shape, action_shape, device, critic_lr, actor_lr, 1)

agent.actor.load_state_dict(torch.load(actor_weights_path))
agent.critic.load_state_dict(torch.load(critic_weights_path))
agent.actor.eval()
agent.critic.eval()

# SHOWCASE AGENT

n_showcase_episodes = 4

for episode in range(n_showcase_episodes):
    print(f"starting episode {episode}...")

    # create a new sample environment to get new random parameters
    env = gym.make("CartPole-v1", render_mode="human", max_episode_steps=500)

    # get an initial state
    state, info = env.reset()

    # play one episode
    done = False
    while not done:
        # select an action A_{t} using S_{t} as input for the agent
        with torch.no_grad():
            action, _, _, _ = agent.select_action(state,bool_greedy=True)

        # perform the action A_{t} in the environment to get S_{t+1} and R_{t+1}
        state, reward, terminated, truncated, info = env.step(action.item())

        # update if the environment is done
        done = terminated or truncated

env.close()