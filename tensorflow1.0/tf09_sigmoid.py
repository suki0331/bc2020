# sigmoid(x) = 1/(1+e^x)

import tensorflow as tf

tf.set_random_seed(777)

x_data = [[1, 2],
          [2, 3],
          [3, 1],
          [4, 3],
          [5, 3],
          [6, 2]]

y_data = [[0],
          [0],
          [0], 
          [1], 
          [1],
          [1]]

x = tf.placeholder(tf.float32, shape=[None, 2])
y = tf.placeholder(tf.float32, shape=[None, 1])

# shape of w is quite important
w = tf.Variable(tf.random_normal([2, 1]), name = 'weight', dtype = tf.float32)
b = tf.Variable(tf.random_normal([1]), name = 'bias',dtype = tf.float32)

hypothesis = tf.sigmoid(tf.matmul(x, w) + b)     # matrix multiplication 

cost = -tf.reduce_mean(y * tf.log(hypothesis) 
                        + (1-y) * tf.log(1-hypothesis))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-2)
train = optimizer.minimize(cost)

predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, y), dtype=tf.float32))


sess = tf.Session()
sess.run(tf.global_variables_initializer())
for step in range(9999):
    cost_val, _ = sess.run([cost, train],
                            feed_dict = {x : x_data, y: y_data})
    # cost_val, hy_val, _ = sess.run([cost, hypothesis, train],
    #                         feed_dict = {x : x_data, y: y_data})
 

    if step % 200 == 0:
        print(step, cost_val)
    h,c,a = sess.run([hypothesis, predicted, accuracy], feed_dict={x:x_data, y:y_data})
    # print(a)
print(a)

sess.close()
