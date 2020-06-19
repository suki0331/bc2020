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
observation = env.reset()   # always reset b4 start

# print(env.action_space)         # Box(1, )
# print(env.observation_space)    # Box(3, )

# print(env.observation_space.high)   #[1. 1. 8.]
# print(env.observation_space.low)    #[-1. -1. -8.]
memory = deque(maxlen=10000)
scores = deque(maxlen=100)
avg_scores = []     # initialization
epsilon = 1.0
gamma = 1.0
epsilon_min = 0.01

x_batch, y_batch = [0,0,0], []


print(observation)
print(observation.shape)
observation = np.reshape(observation, [1, env.observation_space.shape[0]])


def model_1():
    model = Sequential()
    model.add(Dense(24, input_shape=(env.observation_space.shape[0], ),  activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(env.action_space.shape[0], activation='linear'))

    model.compile(loss='mse', optimizer='adam')
    return model



done = False
i = 0


model1 = model_1()

for i_episodes in range(1000):
    observation = env.reset()   # always reset b4 start
    done = False  
    observation = np.reshape(observation, [1, env.observation_space.shape[0]])
    while not done:
        env.render()              
        if np.random.random() <= epsilon:           # Epsilon greedy
            action = env.action_space.sample()
            epsilon = epsilon*0.995                 # decaying epsilon
        # print(env.action_space.sample())
        else:
            observation = np.reshape(observation, [1, env.observation_space.shape[0]])
            action = model1.predict(observation)
            # print(f"a {action} \n r {reward}")
        next_observation, reward, done, _ = env.step(action)
        memory.append((observation, action, reward, next_observation, done))
        # if done == False:
            # print(f"reward : {reward} , done : {done}")
        next_observation = np.reshape(next_observation, [env.observation_space.shape[0]],)  # reshape data
        observation = next_observation
        # print(f"p: {observation.shape}")
        # print(memory.shape)
        if len(memory) > 64:
            minibatch = random.sample(memory, 32)
            # print(minibatch[0][0].shape) # observation (1, 3)
            # print(minibatch[0][2])
            # print(minibatch[1][2])        
            # print(minibatch[0][0])  
            for observation, action, reward, next_observation, done in minibatch:
        #         print(observation.shape)
                # print(f"y_target : {y_target}")
                # minibatchavg =
                # if minibatch_avg < minibatchavg:

                if done == True:
                    y_target[0][1] = reward 
                else:
                    y_target[0][1] = reward + gamma * np.max(model1.predict(next_observation))            
        #         x = np.vstack((x_batch,observation))
                # print(f"x_batch : {x}")
                # y = np.vstack((y_batch,y_target[0]))
            # model1.fit(x, y, batch_size=32, verbose=0)
    # env.close()