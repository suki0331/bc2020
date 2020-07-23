import numpy as np

import matplotlib.pyplot as plt
from keras.utils import  np_utils
import pandas as pd
from keras.datasets import mnist
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

(x_train, y_train), (x_test, y_test) = mnist.load_data()

def next_batch(num, data, labels):
       '''
       Return a total of `num` random samples and labels. 
       '''
       idx = np.arange(0 , len(data))
       np.random.shuffle(idx)
       idx = idx[:num]
       data_shuffle = [data[i] for i in idx]
       labels_shuffle = [labels[i] for i in idx]

       return np.asarray(data_shuffle), np.asarray(labels_shuffle)

# Xtr, Ytr = np.arange(0, 10), np.arange(0, 100).reshape(10, 10)
# print(Xtr)
# print(Ytr)



print(x_train.shape)    # (60000, 28, 28) batch_size = 60000 28 * 28 이미지
print(x_test.shape)     # (10000, 28, 28) batch_size = 10000 28 * 28
print(y_train.shape)    # (60000,)  inputdim = 1
print(y_test.shape)     # (10000,)
# y_train = y_train.reshape(y_train[0],1)
print(x_train.shape[0])
print(x_train.__class__)
x_train = x_train / 255
x_test = x_test / 255

x_train = x_train.reshape((x_train.shape[0], x_train.shape[1]*x_train.shape[2]*1))
x_test = x_test.reshape((x_test.shape[0], x_test.shape[1]*x_test.shape[2]*1))

y_train = np_utils.to_categorical(y_train)
y_test= np_utils.to_categorical(y_test)
print('y_train : ', y_train.shape)

print(x_train.shape)    
print(x_test.shape)     

# Xtr, Ytr = next_batch(100, x_train, y_train)
# print(Xtr.shape)
# print(Ytr.shape)



# plt.imshow(x_train[0], 'gray')
# plt.imshow(x_train[0])
# plt.show()

learning_rate = 0.001
training_epochs = 15
batch_size = 100
total_batch = int(len(x_train) / batch_size) # 60000 / 100


x = tf.compat.v1.placeholder(tf.float32, [None, 784])
y = tf.compat.v1.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32) # dropout

w1 = tf.get_variable("w1", shape=[784,512], dtype=tf.float32,           # get_variable 초기값 알아서 설정, 메모리에서 유리
                     initializer=tf.contrib.layers.xavier_initializer())
print("=============w1=================") # 텐서형이기 때문에 자료형만 나옴
print("w1 : ", w1)      # (784, 512)
b1 = tf.Variable(tf.random.normal([512]))
print("=============b1=================")
print("b1 : ", b1 )     # (512, )
L1 = tf.nn.selu(tf.matmul(x,w1)+b1)
print("=============selu=================")
print(L1)               #(?, 512)
L1 = tf.nn.dropout(L1,keep_prob=keep_prob) 
print("=============Drop=================")
print(L1)               #(?, 512)



w2 = tf.get_variable("w2", shape=[512,512], dtype=tf.float32,          
                     initializer=tf.contrib.layers.xavier_initializer())
b2 = tf.Variable(tf.random_normal([512]))
L2 = tf.nn.selu(tf.matmul(L1,w2)+b2)
L2 = tf.nn.dropout(L2,keep_prob=keep_prob) 



w3 = tf.get_variable("w3", shape=[512,512], dtype=tf.float32,          
                     initializer=tf.contrib.layers.xavier_initializer())
b3 = tf.Variable(tf.random_normal([512]))
L3 = tf.nn.selu(tf.matmul(L2,w3)+b3)
L3 = tf.nn.dropout(L3,keep_prob=keep_prob) 



w4 = tf.get_variable("w4", shape=[512,256], dtype=tf.float32,          
                     initializer=tf.contrib.layers.xavier_initializer())
b4 = tf.Variable(tf.random_normal([256]))
L4 = tf.nn.selu(tf.matmul(L3,w4)+b4)
L4 = tf.nn.dropout(L4,keep_prob=keep_prob) 


w5 = tf.get_variable("w5", shape=[256,10], dtype=tf.float32,          
                     initializer=tf.contrib.layers.xavier_initializer())
b5 = tf.Variable(tf.random_normal([10]))
hypothesis = tf.nn.softmax(tf.matmul(L4,w5)+b5)



cost = tf.reduce_mean(-tf.reduce_sum(y*tf.math.log(hypothesis), axis = 1))




optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate= learning_rate).minimize(cost)
# train = optimizer.minimize(cost)

prediction = tf.equal(tf.math.argmax(hypothesis, 1), tf.arg_max(y,1))
accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))
with tf.compat.v1.Session() as sess: # Session을 close하지 않으려고
    sess.run(tf.compat.v1.global_variables_initializer())


    for epoch in range(training_epochs): # 15
        avg_cost = 0


        for i in range(total_batch):    # 600
            batch_xs, batch_ys = next_batch(100, x_train, y_train)
            feed_dict = {x:batch_xs, y:batch_ys, keep_prob : 1}
            c, _, acc_val= sess.run([cost, optimizer, accuracy], feed_dict=feed_dict)
            avg_cost += c / total_batch
        print("Epoch : ", "%4d" % (epoch+1) , "cost = " , "{:.9f}".format(avg_cost))

            
    feed_dict_test = {x:x_test, y:y_test, keep_prob : 0.7}

    acc = sess.run([accuracy], feed_dict = feed_dict_test)

        
    
    # h,p,a = sess.run([hypothesis,prediction, accuracy], feed_dict={x:x_test, y:y_test})
    print('훈련 끝')
    correct_prediction = tf.equal(tf.argmax(hypothesis,1), tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    _, acc = sess.run([hypothesis,accuracy], feed_dict={x: x_test, y: y_test, keep_prob : 0.6})
    
    print('Acc : ',acc ) ## acc 출력
# 









'''

   for epoch in range(training_epochs): # 15
        avg_cost = 0


        for i in range(0,600,batch_size):    # 600
            batch_xs, batch_ys = x_train[i*batch_size : (i*batch_size) + batch_size], y_train[i*batch_size: (i*batch_size)+ batch_size]
            c, _ , acc_val= sess.run([cost, optimizer, accuracy], feed_dict={x: batch_xs, y: batch_ys,keep_prob : 0.7 })
            avg_cost += c / total_batch

        print("Epoch : ", "%4d" % (epoch+1) , "cost = " , "{:.9f}".format(avg_cost))
        
    
    # h,p,a = sess.run([hypothesis,prediction, accuracy], feed_dict={x:x_test, y:y_test})
    print('훈련 끝')
    correct_prediction = tf.equal(tf.argmax(hypothesis,1), tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    _, pre, acc = sess.run([hypothesis,prediction,accuracy], feed_dict={x: x_test, y: y_test, keep_prob : 0.7})
    
    print('Acc : ',acc ) ## acc 출력





  for i in range(total_batch):    # 600
            batch_xs, batch_ys = next_batch(100, x_train, y_train)
            feed_dict = {x:batch_xs, y:batch_ys, keep_prob : 0.7}
            c, _, acc_val= sess.run([cost, train, accuracy], feed_dict=feed_dict)
            avg_cost += c / total_batch
            
    batch_t_xs, batch_t_ys = next_batch(100, x_test, y_test)
    feed_dict_test = {x:batch_xs, y:batch_ys, keep_prob : 0.7}

    acc = sess.run([accuracy], feed_dict = feed_dict_test)

        
'''
    