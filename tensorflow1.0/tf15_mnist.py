import tensorflow as tf
from keras.datasets import mnist
# from keras.utils import np_utils


(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape)    # (60000, 28, 28)
print(y_train.shape)    # (60000, )
print(x_test.shape)     # (10000, 28, 28)
print(y_test.shape)     # (10000, )

x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# y_train = np_utils.to_categorical(y_train)
# y_test = np_utils.to_categorical(y_test)
# print(y_train.shape)    # (60000, 10)
# print(y_test.shape)     # (10000, 10)

y_train_onehot = tf.one_hot(y_train, depth=10)
y_test_onehot = tf.one_hot(y_test, depth=10)
print(x_train.shape)
print(y_train_onehot.shape)


x = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, x_train.shape[1], x_train.shape[2], x_train.shape[3]])
y = tf.compat.v1.placeholder(dtype=tf.float32, shapes=[None, y_train_onehot.shape[1]])

# 
w = tf.Variable(tf.random.normal([28, 28, 1, 32], name = 'weight'))
b = tf.Variable(tf.random.normal([32]) , name = 'bias')
# activation = relu
layer1 = tf.nn.relu(tf.matmul(x, w) + b)

x = tf.compat.v1.Flatten(layer1)
w2 = tf.compat.v1.Variable(tf.zeros([ , 10]))
b2 = tf.compat.v1.Variable(tf.zeros([10]))
hypothesis = tf.sigmoid(tf.matmul(x, w2) + b2)

# categorical cross entropy
loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis),axis=1)) 

optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
predicted = tf.argmax(hypothesis , 1)
acc = tf.reduce_mean(tf.cast(tf.equal(predicted, tf.argmax(y_test,1)), dtype=tf.float32))

with tf.Session as sess:
    sess.run(tf.global_variables_initializer())
    
    for step in range(2001):
        _, cost_val = sess.run([optimizer, loss], feed_dict={x:x_train, y:y_train_onehot})

        if step % 200 == 0:
            print(step, cost_val)
    
    result = sess.run([hypothesis, predicted, acc], feed_dict={x:x_test})
    print(result)