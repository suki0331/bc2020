from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
import tensorflow as tf

# regression
dataset = load_diabetes()
x_data = dataset.data
y_data = dataset.target

print(x_data.shape)  # (442, 10)
y_data = y_data.reshape(442,1)
print(y_data.shape)  # (442, )

x_train, x_test, y_train, y_test = train_test_split(
    x_data, y_data, train_size = 0.8
)

print(x_train.shape)    # (353, 10)
print(x_test.shape)     # (89, 10)
print(y_train.shape)    # (353, 1)
print(y_test.shape)     # (89, 1)



x = tf.placeholder(dtype=tf.float32, shape=[None, 10]) 
y = tf.placeholder(dtype=tf.float32, shape=[None, 1])

# linear regression
w = tf.Variable(tf.random_normal([10,1]), name='weight', dtype=tf.float32)
b = tf.Variable(tf.random_normal([1]), name='bias', dtype=tf.float32)

hypothesis = tf.matmul(x,w)+b

cost  = tf.reduce_mean(tf.square(hypothesis-y))

train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(2001):
        train_cost, train_hypo, _ = sess.run([cost, hypothesis, train],
                        feed_dict={x:x_train, y:y_train})
        if step%20 ==0:
            print(train_cost)
    cost_val, hypo_val, _ = sess.run([cost, hypothesis, train], feed_dict={x:x_test, y:y_test})
    print(cost_val, hypo_val)
sess.close()