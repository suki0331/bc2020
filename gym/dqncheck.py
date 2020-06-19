import tensorflow as tf
import numpy as np
from collections import deque
import gym
import math
import random
from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dense
from keras.optimizers import Adam
from gym import wrappers

env_name = 'Pendulum-v0'
env = gym.make(env_name)
env = wrappers.Monitor(env, './recording_real1', force=True)
memory = deque(maxlen=10000)
gamma = 0.9


print(env.action_space)         # Box(1,)
print(env.observation_space)    # Box(3,)

print(env.observation_space.shape[0])
# print(env.action_space.shape[0])

# print(env.observation_space.high)   #[1. 1. 8.]
# print(env.observation_space.low)    #[-1. -1. -8.]



def model_1():
    model = Sequential()
    model.add(Dense(24, input_shape=(env.observation_space.shape[0], ),  activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(env.action_space.shape[0], activation='linear'))

    model.compile(loss='mse', optimizer='adam')
    return model


# def model_fit(model, x, y):
#     return model.fit(x, y, verbose=0)

def preprocess(observation):
    return np.reshape(observation, [-1, env.observation_space.shape[0]])

def model_train(model, minibatch):
    x_stack = np.empty(0).reshape(0, 3)
    y_stack = np.empty(0).reshape(0, 1)

    for observation, action, reward, next_observation, done in minibatch:
        Q = model.predict(observation)

        if done:
            Q[0,0] = reward
        else:
            Q[0,0] = reward + gamma*np.max(model.predict(next_observation))
        
        y_stack = np.vstack([y_stack, Q])
        x_stack = np.vstack([x_stack, observation])
    return model.fit(x_stack,y_stack, batch_size=128, verbose=2)




model = model_1()

for i_episode in range(100000):
    observation = env.reset()   # always reset b4 start
    done = False
    observation = preprocess(observation)
    epsilon = 1.0
    epsilon_min = 0.01
    reward_sum = 0
    while True:
        env.render()
        observation = preprocess(observation)
        if np.random.random() <= max(epsilon, epsilon_min):
            action = env.action_space.sample()
        else:    
            action = model.predict(observation)
        epsilon = epsilon*0.98
        next_observation, reward, done, info = env.step(action)
        next_observation = preprocess(next_observation)
        memory.append((observation, action, reward, next_observation, done))
        observation = next_observation
        reward_sum += reward
        if done:
            print(f"episode {i_episode} Total reward : {reward_sum}")
            break
    # observation = np.reshape(observation, [1, env.observation_space.shape[0]])
    # next_observation = np.reshape(next_observation, [1, env.observation_space.shape[0]])
    # print(observation.shape)
    # print(next_observation.shape)
    minibatch = random.sample(memory, 128)
    model_train(model, minibatch)
    reward += reward_sum
# env.close()