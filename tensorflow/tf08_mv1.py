import tensorflow as tf
import matplotlib.pyplot as plt

tf.set_random_seed(1)

x1_data = [73, 93, 89, 96, 73]
x2_data = [80, 88, 91, 98, 66]
x3_data = [75, 93, 90, 100, 70]

y_data = [152, 185, 180, 196, 142]

x1 = tf.placeholder(dtype=tf.float32, shape=[None])
x2 = tf.placeholder(dtype=tf.float32, shape=[None])
x3 = tf.placeholder(dtype=tf.float32, shape=[None])
y = tf.placeholder(dtype=tf.float32, shape=[None])

w1 = tf.Variable(tf.random_normal([1]), name='weight1', dtype=tf.float32)
w2 = tf.Variable(tf.random_normal([1]), name='weight2', dtype=tf.float32)
w3 = tf.Variable(tf.random_normal([1]), name='weight3', dtype=tf.float32)

b = tf.Variable(tf.random_normal([1]), name='bias1', dtype=tf.float32)

hypothesis = x1*w1 + x2*w2 + x3*w3 + b

cost = tf.reduce_mean(tf.square(hypothesis-y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.00005)
train = optimizer.minimize(cost)

# sess = tf.Session()
# sess.run(tf.global_variables_initializer())
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(2001):
        a = sess.run([cost, hypothesis, train],
                                        feed_dict={x1:x1_data, x2:x2_data, x3:x3_data, y:y_data})
        # cost_val, hy_val, _ = sess.run([cost, hypothesis, train],
        #                                 feed_dict={x1:x1_data, x2:x2_data, x3:x3_data, y:y_data})
        if step % 10 == 0:
            print(a)
            # print(cost_val, hy_val)

sess.close()