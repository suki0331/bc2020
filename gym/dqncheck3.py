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

env_name = 'CartPole-v0'
env = gym.make(env_name)
# env = wrappers.Monitor(env, './recording_real2', force=True)
memory = deque(maxlen=3000)
gamma = 0.9


print(env.action_space)         # Box(1,)
print(env.observation_space)    # Box(3,)

print(env.observation_space.shape[0])
# print(env.action_space.shape[0])

# print(env.observation_space.high)   #[1. 1. 8.]
# print(env.observation_space.low)    #[-1. -1. -8.]



def model_1():
    model = Sequential()
    model.add(Dense(12, input_shape=(env.observation_space.shape[0], ),  activation='relu'))
    model.add(Dense(12, activation='relu'))
    model.add(Dense(env.action_space.n, activation='linear'))

    model.compile(loss='mse', optimizer=Adam(lr=0.001))
    return model


# def model_fit(model, x, y):
#     return model.fit(x, y, verbose=0)

def preprocess(observation):
    return np.reshape(observation, [-1, env.observation_space.shape[0]])

def model_train(model, minibatch):
    x_stack = np.empty(0).reshape(0, 4)
    y_stack = np.empty(0).reshape(0, 2)

    for observation, action, reward, next_observation, done in minibatch:
        Q = model.predict(observation)
        
        if done:
            Q[0,action] = reward
        else:
            Q[0,action] = reward + gamma*np.max(np.argmax(model.predict(next_observation)))
        
        y_stack = np.vstack([y_stack, Q])
        x_stack = np.vstack([x_stack, observation])
    return model.fit(x_stack,y_stack, epochs=1, batch_size=32, verbose=0)



model = model_1()


for i_episode in range(100000):
    observation = env.reset()   # always reset b4 start
    done = False
    observation = preprocess(observation)
    epsilon = 1       # /(i_episode*0.001+1)
    epsilon_min = 0.01
    reward_sum = 0
    while True:
        # env.render()
        if np.random.random() <= max(epsilon, epsilon_min):
            action = env.action_space.sample()
        else:    
            observation = preprocess(observation)
            action = np.argmax(model.predict(observation))
        epsilon = epsilon*0.99
        next_observation, reward, done, info = env.step(action)
        next_observation = preprocess(next_observation)
        memory.append((next_observation, action, reward, observation, done))
        reward_sum += reward
        if done:
            if reward >= 100:
                print(f"episode {i_episode} Total reward : {reward_sum}")
            reward -= 200
            break

        if len(memory) >= 32:
            minibatch = random.sample(memory, 32)
            model_train(model, minibatch)
        observation = next_observation
    # observation = np.reshape(observation, [1, env.observation_space.shape[0]])
    # next_observation = np.reshape(next_observation, [1, env.observation_space.shape[0]])
    # print(observation.shape)
    # print(next_observation.shape)
# env.close()