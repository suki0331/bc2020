# x, y, w, b, hypothesis, cost, train
# sigmoid 사용
# predict, accuracy 준비해 놓을 것

import tensorflow as tf
import numpy as np

tf.set_random_seed(777)

x_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
y_data = np.array([[0], [1], [1], [0]], dtype=np.float32)

print(x_data.shape)     # (4, 2)
print(y_data.shape)     # (4, 1)


x = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None,2])
y = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None,1])

# 1-layer
w1 = tf.Variable(tf.random_normal([2, 100]), name='weight1')
b1 = tf.Variable(tf.random_normal([100]),    name='bias1')
layer1 = tf.sigmoid(tf.matmul(x, w1) + b1)

# 2-layer
w2 = tf.Variable(tf.random_normal([100, 50]), name='weight2')
b2 = tf.Variable(tf.random_normal([50]),    name='bias2')
layer2 = tf.sigmoid(tf.matmul(layer1, w2) + b2)

# 3-layer
w3 = tf.Variable(tf.random_normal([50, 1]), name= 'weight3')
b3 = tf.Variable(tf.random_normal([1]), name ='bias3')
hypothesis = tf.sigmoid(tf.matmul(layer2, w3) + b3)


cost = -tf.reduce_mean(y * tf.log(hypothesis) + (1 - y) *tf.log(1 - hypothesis))


optimizer = tf.train.GradientDescentOptimizer(learning_rate = 1e-2)
train = optimizer.minimize(cost)


predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)


accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, y), dtype=tf.float32))


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(2001):
        cost_val, _ = sess.run([cost, train], feed_dict={x:x_data, y:y_data})
        
        if step % 20 == 0:
            print(step, cost_val)

    h, c, a = sess.run([hypothesis, predicted, accuracy], feed_dict={x:x_data, y:y_data})
    print("\n\n Hypothesis :", "\n",h, "\n\n Predict :","\n",c, "\n\n accuracy :", a)