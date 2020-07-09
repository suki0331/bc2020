import tensorflow as tf

tf.set_random_seed(777)

x_data = [[73, 51, 65],
          [92, 98, 11],
          [89, 31, 33],
          [99, 33, 100],
          [17, 66, 79]]

y_data = [[152],
          [185],
          [180], 
          [196], 
          [142]]

x = tf.placeholder(tf.float32, shape=[None, 3])
y = tf.placeholder(tf.float32, shape=[None, 1])

# shape of w is quite important
w = tf.Variable(tf.random_normal([3, 1]), name = 'weight', dtype = tf.float32)
b = tf.Variable(tf.random_normal([1]), name = 'bias', dtype = tf.float32)

hypothesis = tf.matmul(x, w) + b     # matrix multiplication 

cost = tf.reduce_mean(tf.square(hypothesis-y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
for step in range(999901):
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train],
                            feed_dict = {x : x_data, y: y_data})
 

    if step % 10 == 0:
        print(f"step : {step},  cost : {cost_val}, hy_val : {hy_val}")

sess.close()