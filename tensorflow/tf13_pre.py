import tensorflow as tf
import numpy as np

def min_max_scaler(dataset):
    numerator = dataset - np.min(dataset, 0)
    denominator = np.max(dataset,0) - np.min(dataset, 0)
    return numerator / (denominator + 1e-7)

dataset = np.array([[828.659973, 833.450012, 908100, 828.349976, 831.659973],
                    [823.02002, 828.070007, 1828100, 821.655029, 828.070007],
                    [819.929993, 824.400024, 1438100, 818.97998, 824.159973],
                    [816, 820.958984, 1008100, 815.48999, 819.23999],
                    [819.359985, 823, 1188100, 818.469971, 818.97998],
                    [819, 823, 1198100, 816, 820.450012],
                    [811.700012, 815.25, 1098100, 809.780029, 813.669983],
                    [809.51001, 816.659973, 1398100, 804.539978, 809.559998]])

dataset = min_max_scaler(dataset)
print(dataset)

x_data = dataset[:, 0:-1]
y_data = dataset[:,[-1]]

print(x_data.shape) # (8, 4)
print(y_data.shape) # (8, 1)


x = tf.placeholder(tf.float32, shape=[None, 4])
y = tf.placeholder(tf.float32, shape=[None, 1])

w = tf.Variable(tf.random_normal([4, 1]), name='Weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = tf.matmul(x, w) + b   

cost = tf.reduce_mean(tf.square(hypothesis - y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.0099)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())



for step in range(10001):
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train], feed_dict={x:x_data, y:y_data})

    if step % 200 == 0:
        print(step, "cost:", cost_val, "\n 실제값:",y_data, "예측값:", hy_val)