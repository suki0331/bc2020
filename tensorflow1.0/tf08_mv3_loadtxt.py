import math
import numpy as np
import pandas as pd
import tensorflow as tf
tf.set_random_seed(777)


dataset = np.loadtxt('d:/study/data/csv/data-01-test-score.csv'
                    , delimiter=',', dtype=np.float32)
# print(dataset)

x_data = dataset[:, :-1]
y_data = dataset[:, -1]
x_data = x_data.reshape(-1, 3)
y_data = y_data.reshape(-1, 1)
print(x_data.shape)
print(y_data.shape)
print(x_data)
print(y_data)

x = tf.placeholder(tf.float32, shape=[None, 3])
y = tf.placeholder(tf.float32, shape=[None, 1])

# shape of w is quite important
w = tf.Variable(tf.random_normal([3, 1]), name = 'weight', dtype = tf.float32)
b = tf.Variable(tf.random_normal([1]), name = 'bias',dtype = tf.float32)

hypothesis = tf.matmul(x, w) + b     # matrix multiplication 

cost = tf.reduce_mean(tf.square(hypothesis-y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=math.inf)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
for step in range(999901):
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train],
                            feed_dict = {x : x_data, y: y_data})
 

    if step % 10 == 0:
        print(f"step : {step},  cost : {cost_val}, hy_val : {hy_val}")

sess.close()
