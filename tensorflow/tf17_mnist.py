import tensorflow as tf
from keras.datasets import mnist
# from keras.utils import np_utils


(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape)    # (60000, 28, 28)
print(y_train.shape)    # (60000, )
print(x_test.shape)     # (10000, 28, 28)
print(y_test.shape)     # (10000, )

x_train = x_train.reshape(-1, 784)
x_test = x_test.reshape(-1, 784)

# y_train = np_utils.to_categorical(y_train)
# y_test = np_utils.to_categorical(y_test)
# print(y_train.shape)    # (60000, 10)
# print(y_test.shape)     # (10000, 10)

y_train_onehot = tf.one_hot(y_train, depth=10)
y_test_onehot = tf.one_hot(y_test, depth=10)
print(x_train.shape)
print(y_train_onehot.shape)

LEARNING_RATE = 0.001
TRAINING_EPOCHS = 15
BATCH_SIZE = 100
TOTAL_BATCH =int(len(x_train) / BATCH_SIZE)

x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)  # dropout


w1 = tf.get_variable("w1", shape=[784,512],
        initializer=tf.contrib.layers.xavier_initializer())
b1 = tf.Variable(tf.random.normal([512]))
l1 = tf.nn.selu(tf.matmul(x, w1)+b1)
l1 = tf.nn.dropout(l1, keep_prob=keep_prob)


w2 = tf.get_variable("w2", shape=[512,512],
        initializer=tf.contrib.layers.xavier_initializer())
b2 = tf.Variable(tf.random.normal([512]))
l2 = tf.nn.selu(tf.matmul(l1, w2)+b2)
l2 = tf.nn.dropout(l2, keep_prob=keep_prob)

w3 = tf.get_variable("w3", shape=[512,128],
        initializer=tf.contrib.layers.xavier_initializer())
b3 = tf.Variable(tf.random.normal([128]))
l3 = tf.nn.selu(tf.matmul(l2, w3)+b3)
l3 = tf.nn.dropout(l3, keep_prob=keep_prob)


w4 = tf.get_variable("w4", shape=[128,10],
        initializer=tf.contrib.layers.xavier_initializer())
b4 = tf.Variable(tf.random.normal([10]))
hypothesis = tf.nn.softmax(tf.matmul(l3, w4)+b4)

cost = tf.reduce_mean(-tf.reduce_sum(y*tf.math.log(hypothesis),axis=1))
optimizer =  tf.compat.v1.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(TRAINING_EPOCHS):
        ave_cost = 0
        
        for i in range(TOTAL_BATCH):
            batch_xs, batch_ys