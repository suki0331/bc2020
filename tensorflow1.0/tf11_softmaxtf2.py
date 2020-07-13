import tensorflow as tf
import numpy as np

seed = 77

tf.compat.v1.set_random_seed(seed)

x_data = np.array([[1, 2, 1, 1],
          [2, 1, 3, 2],
          [3, 1, 3, 4],
          [4, 1, 5, 5],
          [1, 7, 5, 5],
          [1, 2, 5, 6],
          [1, 6, 6, 6],
          [1, 7, 6, 7]], dtype=np.float32)

y_data = np.array([[0, 0, 1],
          [0, 0, 1],
          [0, 0, 1],
          [0, 1, 0],
          [0, 1, 0],
          [0, 1, 0],
          [1, 0, 0],
          [1, 0, 0]], dtype=np.float32)

print(x_data.shape)
print(y_data.shape)

x_col_num = x_data.shape[1] # 4
y_col_num = y_data.shape[1] # 3

x = tf.compat.v1.placeholder(tf.float32, shape=[None, x_col_num])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, y_col_num])

w = tf.Variable(tf.random.normal([4, 3]), name = 'weight')
b = tf.Variable(tf.random.normal([1, 3]), name = 'bias') # y_col_num

h = tf.nn.softmax(tf.matmul(x,w) + b)

loss = tf.reduce_mean(input_tensor=-tf.reduce_sum(input_tensor=y * tf.math.log(h),axis=1)) # loss

opt = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=1e-4).minimize(loss) #

# train = opt.minimize(loss)

alldt = {x:[[1,11,7,9],[11,1,7,9], [11,13,7,9], [10,21,17,9]]}

with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())

    for i in range(2001):
        _, cost_val = sess.run([opt, loss], feed_dict={x: x_data, y: y_data})

        if i % 200 == 0 :
            print( i , cost_val)
    
    a = sess.run(h, feed_dict={x:[[1,11,7,9]]})
    print(a, sess.run(tf.argmax(input=a, axis=1)))

    b = sess.run(h, feed_dict={x:[[11,1,7,9]]})
    print(b, sess.run(tf.argmax(input=b, axis=1)))

    c = sess.run(h, feed_dict={x:[[11,13,7,9]]})
    print(c, sess.run(tf.argmax(input=c, axis=1)))

    d = sess.run(h, feed_dict={x:[[10,21,17,9]]})
    print(d, sess.run(tf.argmax(input=d, axis=1)))

    print('')

    all = sess.run(h, feed_dict=alldt)
    print(all, sess.run(tf.argmax(input=all,axis=1)))


    # feed_dict={x: [np.append(a, 0), np.append(b, 0), np.append(c, 0)]})