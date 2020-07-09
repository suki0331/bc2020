# from tf06_1.py
# overwrite learning rate number
# 0.01 >> 0.1 / 0.001 1
# overwrite num of epochs

import tensorflow as tf

tf.set_random_seed(777)


x_train = tf.placeholder(dtype=tf.float32, shape=[None])
y_train = tf.placeholder(dtype=tf.float32, shape=[None])

# x_train = [1,2,3]
# y_train = [3,5,7]

# feed_dict = {x_train_hold:x_train, y_train_hold:y_train}


W = tf.Variable(tf.random_normal([1]), name='Weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# sess = tf.Session()
# sess.run(tf.global_variables_initializer()) 
# print(sess.run(W))

hypothesis = x_train * W + b       

cost = tf.reduce_mean(tf.square(hypothesis - y_train))  

train = tf.train.GradientDescentOptimizer(learning_rate=0.015).minimize(cost) 

with tf.Session() as sess:  
    sess.run(tf.global_variables_initializer())

    for step in range(2001):    
        _, cost_val, W_val, b_val = sess.run([train, cost, W, b], feed_dict={x_train:[1, 2, 3], y_train:[3, 5, 7]})  

        if step % 20 == 0:  
            print(step, cost_val, W_val, b_val)
    
    print('예측: ', sess.run(hypothesis, feed_dict={x_train:[4]}))         # 예측:  [9.000078]
    print('예측: ', sess.run(hypothesis, feed_dict={x_train:[5, 6]}))      # 예측:  [11.000123 13.000169]
    print('예측: ', sess.run(hypothesis, feed_dict={x_train:[6, 7, 8]}))