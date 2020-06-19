import tensorflow as tf
import numpy as np
from collections import deque
import gym
import math
import random
from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dense
from keras.optimizers import Adam

env_name = 'Pendulum-v0'
env = gym.make(env_name)

env.reset()
print(env.action_space)         # Discrete(2)
print(env.observation_space)    # Box(4,)

print(env.observation_space.shape[0])
# print(env.action_space.shape[0])
print(env.reward_range)
# print(env.observation_space.high)   #[1. 1. 8.]
# print(env.observation_space.low)    #[-1. -1. -8.]





observation = env.reset()   # always reset b4 start
done = False
while True:
    env.render()
    action = env.action_space.sample()
    # print(observation)
    observation, reward, done, info = env.step(action)
    # print(f"action : {action}, reward : {reward}")
    # if reward >= -0.2:
    #     print("rr")
    print(reward)        

# env.close()
