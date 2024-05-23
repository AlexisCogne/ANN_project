import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from A2C_discrete import A2C
from tqdm import tqdm
import os
import time
import mujoco

print(mujoco.__version__)
# create a new sample environment to get new random parameters
env = gym.make("InvertedPendulum-v4",render_mode = "human",max_episode_steps=1000) ## Continuous case
states = []
# get an initial state
state, info = env.reset(seed=42)
states.append(state)
done = False
# play one episode
for i in range(6):
    # perform the action A_{t} in the environment to get S_{t+1} and R_{t+1}
    state, reward, terminated, truncated, info = env.step([-0.2]) # only pushes left (0 is left, 1 is right)
    states.append(state)
    #pause for 0.5s
    time.sleep(0.2)    
while not done:
    # perform the action A_{t} in the environment to get S_{t+1} and R_{t+1}
    state, reward, terminated, truncated, info = env.step([1]) # only pushes left (0 is left, 1 is right)
    states.append(state)
    time.sleep(0.5)    
    done = terminated or truncated
    
env.close()
print(len(states))