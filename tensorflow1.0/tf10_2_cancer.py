from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import tensorflow as tf

# regression
dataset = load_breast_cancer()
x_data = dataset.data
y_data = dataset.target

print(x_data.shape)  # (569, 30)
print(y_data.shape)  # (569, )
y_data = y_data.reshape(-1, 1)

x_train, x_test, y_train, y_test = train_test_split(
    x_data, y_data, train_size = 0.8
)

print(x_train.shape)    # (455, 30)
print(x_test.shape)     # (114, 30)
print(y_train.shape)    # (455, 1)
print(y_test.shape)     # (114, 1)


x = tf.placeholder(dtype=tf.float32, shape=[None, 30]) 
y = tf.placeholder(dtype=tf.float32, shape=[None, 1])

# logistic regression
w = tf.Variable(tf.zeros([30,1]), name='weight', dtype=tf.float32)
b = tf.Variable(tf.zeros([1]), name='bias', dtype=tf.float32)

hypothesis = tf.sigmoid(tf.matmul(x,w)+b)

cost  = -tf.reduce_mean(y*tf.log(hypothesis) + (1-y)*tf.log(1-hypothesis))

train = tf.train.GradientDescentOptimizer(learning_rate=0.5e-6).minimize(cost)

predicted = tf.cast(hypothesis >= 0.5, dtype=tf.float32)

accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted,y), dtype=tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(2001):
        train_cost, _ = sess.run([cost, train],
                        feed_dict={x:x_train, y:y_train})
        if step%20 ==0:
            print(step, train_cost)
    cost_val, hypo_val, acc = sess.run([hypothesis, predicted, accuracy], feed_dict={x:x_test, y:y_test})
    print(cost_val, hypo_val, acc)
sess.close()