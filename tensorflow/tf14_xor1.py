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


x = tf.placeholder(dtype=float, shape=[None,2])
y = tf.placeholder(dtype=float, shape=[None,1])

w = tf.Variable(tf.random_normal([2, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]),    name='bias')

# hypoth = tf.matmul(w, x) + b
# sig_hy = tf.sigmoid(hypoth)

# loss = -tf.reduce_mean(y * tf.log(sig_hy) + (1 - y) *tf.log(1 - sig_hy))

# otm = tf.train.GradientDescentOptimizer(learning_rate=1e-7)
# train = otm.minimize(loss)

# predicted = tf.cast(sig_hy > 0.5, dtype=tf.float32)

# accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, y), dtype=tf.float32))

# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())

#     for step in range(2001):
#         ls_val,  _ = sess.run([loss, train], feed_dict={x:x_data, y:y_data})

#     if step % 20 == 0:
#         print(step, "loss:", ls_val)

#     h, c, a = sess.run([sig_hy, predicted, accuracy], feed_dict={x:x_data, y:y_data})
#     print("\n\n Hypothesis :", "\n",h, "\n\n Predict :","\n",c, "\n\n accuracy :", a)



hypothesis = tf.sigmoid(tf.matmul(x, w) + b)

cost = -tf.reduce_mean(y * tf.log(hypothesis) + (1 - y) *tf.log(1 - hypothesis))

optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.5e-6)
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