import tensorflow as tf
import numpy as np
from collections import deque
import gym
import math
import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam

EPOCHS= 1000
THRESHOLD = 45
MONITOR = True

class DQN:
    
    def __init__(self, env_string, batch_size=64, IM_SIZE = 84, m = 4):
        self.memory = deque(maxlen=100000)
        self.env = gym.make(env_string)
        self.input_size = self.env.observation_space.shape[0]
        action_size = self.env.action_space.n
        batch_size = self.batch_size
        self.gamma = 1.0
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.IM_SIZE = IM_SIZE
        self.m = m

        alpha = 0.01
        alpha_decay = 0.01
        if MONITOR: self.env = gym.wrappers.Monitor(self.env, '../data/'+env_string, force=True)
  
        # initialize model
        self.model = Sequential()
        self.model.add(Conv2D(32, 8, (4,4), activation='relu', padding='same', input_shape=(IM_SIZE, IM_SIZE, m)))
        self.model.add(Conv2D(64, 4, (2,2), activation='relu', padding='valid'))
        self.model.add(Conv2D(64, 3, (1,1), activation='relu', padding='valid'))
        self.model.add(Flatten())
        self.model.add(Dense(512, activation='elu'))
        self.model.add(Dense(action_size, activation='linear'))
        self.model.compile(loss='mse', optimizer=Adam(lr=alpha, decay=alpha_decay))

    def combine_images(self, img1, img2):
        if len(img1.shape) == 3 and img1.shape[0] == self.m:
            im = np.append(img1[1:,:, :], np.expand_dims(img2,0), axis=2)
            return tf.expand_dims(im, 0)
        else:
            im = np.stack([img1]*self.m, axis=2)
            return tf.expand_dims(im, 0)
        
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        x_batch, y_batch = [], []
        minibatch = random.sample(self.memory, min(len(self.memory), batch_size))


        for state, action, reward, next_state, done in minibatch:
            y_target = self.model.predict(state)
            y_target[0][action] = reward if done else reward + self.gamma*np.max(self.model.predict(next_state)[0])
            x_batch.append(state[0])
            y_batch.append(y_target[0])

        self.model.fit(np.array(x_batch), np.array(y_batch), batch_size=len(x_batch), verbose=0)

    def choose_action(self, state, epsilon):
        if np.random.random() <= epsilon:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.model.predict(state))

    def train(self):
        scores = deque(maxlen=100)
        avg_scores = []
        for e in range(EPOCHS):
            state = self.env.reset()
            state = self.preprocess_state(state)
            state = self.combine_images(state, state)
            done = False            
            i = 0
            while not done:
                action = self.choose_action(state, self.epsilon)
                next_state, reward, done, _ = self.env.step(action)
                next_state = self.preprocess_state(next_state)
                next_state = self.combine_images(next_state, state)
                self.remember(state, action, reward, next_state, done)
                state = next_state
                self.epsilon = max(self.epsilon_min, self.epsilon_decay*self.epsilon)   # decay epsilon
                i += reward
            scores.append(i)
            mean_score = np.mean(scores)
            avg_scores.append(mean_score)
            if mean_score >= THRESHOLD and e >= 100:
                print(f"Ran {e} episodes. Solved after {e - 100} trials?")
                return avg_scores
            if e % 100 == 0:
                print(f"[Episode {e}] - Score over last 100 episodes was {mean_score}")
                self.replay(self.batch_size)
                return avg_scores
            print(f"Didn't solve after {e} episodes :")
            return avg_scores

    def preprocess_state(self, img):
        img_temp = img[31:195]  # choose important area of the image
        img_temp = tf.image.rgb_to_grayscale(img_temp)
        img_temp = tf.image.resize(img_temp, [self.IM_SIZE, self.IM_SIZE],
        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        img_temp = tf.cast(img_temp, tf.float32)
        return img_temp[:,:,0]

env_string = 'Breakout-v0'
agent = DQN(env_string)
scores = agent.train()
