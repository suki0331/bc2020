import gym
import random
from collections import deque
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten
from tensorflow.keras.optimizers import Adam



class DQN:
    
    
    
    
    
    def __init__(self, env_string, batch_size = 64):
        self.memory = deque(maxlen=100000)  # use deque to update recent changes 
        self.env = gym.make(env_string)     # access the attributes of the class by using "self"
        self.input_size = self.env.observation_space.shape[0]   # size of input. observation_space == box(4, )
        self.action_size = self.env.action_space.n  # action_space == discrete(2)  
        self.batch_size = batch_size
        self.gamma = 1.0    # initial reward value
        self.epsilon = 1.0  # epsilon value for decaying epsilon-greedy method
        self.epsilon_min = 0.01     # minimal random noise
        # self.epsilon_decay = 0.995  
        
        alpha = 0.01    # learning rate of optimizer
        alpha_decay = 0.01

        # build model
        self.model = Sequential()
        self.model.add(Conv2D(32, (4,4), padding='same', activation='relu'))
        self.model.add(Conv2D(64, (2,2), padding='same', activation='relu'))
        self.model.add(Flatten())
        self.model.add(Dense(512, activation='relu'))
        self.model.add(Dense(action_size, activation='relu'))

        self.model.compile(loss='mse', optimizer=Adam(lr=alpha, decay=alpha_decay))

        def remember(self, state, action, reward, next_state, done):
            self.memory.append((state, action, reward, next_state, done))

        def replay(self, batch_size):
            x_batch, y_batch = [], []
            minibatch = random.sample(self.memory, min)