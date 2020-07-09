# 다중분류 
# iris 코드를 완성하시오.

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import tensorflow as tf

tf.set_random_seed(777)

dataset = load_iris()

print(dataset)

x_data = dataset.data
y_data = dataset.target

print(x_data.shape)  # (150, 4)
print(y_data.shape)  # (150, )
# print(y)
# print(x)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
y_data_onehot = tf.one_hot(y_data, depth=3).eval(session=sess)
print(y_data_onehot)
print(y_data_onehot. shape)   # (150, 3)

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data_onehot, test_size=0.2, shuffle=True, random_state=22)
print(x_train.shape)    # (120, 4)
print(x_test.shape)     # (30, 4)
print(y_train.shape)    # (120, 3)
print(y_test.shape)     # (30, 3)



x = tf.placeholder(tf.float32, shape=[None, 4])
y = tf.placeholder(tf.float32, shape=[None, 3])


w = tf.Variable(tf.random_normal([4, 3]), name = 'weight')
b = tf.Variable(tf.random_normal([1, 3]), name = 'bias')


# aaa = tf.matmul(x,w)+b
# hypothesis = tf.nn.softmax(aaa)
test_hypho = tf.nn.softmax(tf.matmul(x,w) + b)

loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(test_hypho),axis=1)) 
# loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(hypho, y))  


opt = tf.train.GradientDescentOptimizer(learning_rate=1e-4).minimize(loss)

# predicted = tf.cast(tf.argmax(hypho, dtype=tf.float32))
# accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, y), dtype=tf.float32))

predicted = tf.arg_max(test_hypho,1)
acc = tf.reduce_mean(tf.cast(tf.equal(predicted, tf.argmax(y,1)), dtype=tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(2001):
        _, cost_val = sess.run([opt, loss], feed_dict={ x: x_train, y: y_train})

        if step % 200 == 0 :
            print( step , cost_val)

    h, c, a = sess.run([test_hypho, predicted, acc], feed_dict={x:x_test})
    print("\n\n Hypothesis :", "\n",h, "\n\n Predict :","\n",c, "\n\n accuracy :", a)

    # pred = sess.run(hypho, feed_dict={x:x_test})
    # print(pred, sess.run(tf.argmax(pred, 1)))