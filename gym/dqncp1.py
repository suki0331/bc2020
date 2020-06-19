import tensorflow as tf
import numpy as np
from collections import deque
import gym
import math
import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam

env_name = 'CartPole-v0'
env = gym.make(env_name)
observation = env.reset()   # always reset b4 start

# print(env.action_space)         # Discrete(2)
# print(env.observation_space)    # Box(3, )

# print(env.observation_space.high)   #[1. 1. 8.]
# print(env.observation_space.low)    #[-1. -1. -8.]

memory = deque(maxlen=10000)
scores = deque(maxlen=100)
avg_scores = []     # initialization
epsilon = 1.0
gamma = 1.0
epsilon_min = 0.01



print(observation)
print(observation.shape)
observation = np.reshape(observation, [1, env.observation_space.shape[0]])


# model
model = Sequential()
model.add(Dense(24, input_shape=(env.observation_space.shape[0], ),  activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(env.action_space.shape, activation='relu'))

model.compile(loss='mse', optimizer=Adam(lr=0.01, decay=0.01))

done = False
i = 0
env.render()                

while True:
    if np.random.random() <= epsilon:           # Epsilon greedy
        action = env.action_space.sample()
    else:
        action = model.predict(observation)
    epsilon = epsilon*0.995     # decaying epsilon
    next_observation, reward, done, _ = env.step(action)
    print(f"reward : {reward} , done : {done}")
    next_observation = np.reshape(next_observation, [1, env.observation_space.shape[0]])  # reshape data
    observation = next_observation
    memory.append((observation, action, reward, next_observation, done))
    if len(memory) > 64:
        minibatch = random.sample(memory, 64)
        # print(minibatch[0][0].shape) # observation (1, 3)
        # print(minibatch[reward])
        # print(minibatch[0][0])  
        for observation, action, reward, next_observation, done in minibatch:
            # np.reshape(observation, 1 , 3)
            y_target = model.predict(observation)
            print(f"y_target : {y_target}")
            if done == True:
                y_target[0][0] = reward 
            else:
                y_target[0][0] = reward + gamma * np.max(model.predict(next_observation))            
            x = np.vstack((x_batch,observation))
            print(f"x_batch : {x}")
            y = np.vstack((y_batch,y_target[0]))
        model.fit(x, y, batch_size=64, verbose=0)

