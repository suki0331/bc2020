import tensorflow as tf
tf.set_random_seed(777)

x_data =    [[1, 2],
            [2, 3],
            [3, 1],
            [4, 3],
            [5, 3],
            [6, 2]]

y_data = [[0],
         [0],
         [0], 
         [1],
         [1],
         [1]]

x = tf.placeholder(tf.float32, shape=[None, 2])
y = tf.placeholder(tf.float32, shape=[None, 1])


w = tf.Variable(tf.random_normal([2,1]), name = 'weight', dtype = tf.float32)
b = tf.Variable(tf.random_normal([1]), name = 'bias',dtype = tf.float32)

hypothesis = tf.sigmoid(tf.matmul(x,w) + b)

# tf.matmul(x,w) => w*x  x => (5.3)  w => (3,1) w*x => (5,1) 행렬연산을 도와줌

cost = -tf.reduce_mean(y * tf.log(hypothesis) + (1-y)* tf.log(1-hypothesis))


optimizer = tf.train.GradientDescentOptimizer(learning_rate= 1e-2)
train = optimizer.minimize(cost)

predicted = tf.cast(hypothesis >=0.5, dtype=tf.float32)
'''
tf.cast 0.5 보다 크면 true 0.5 보다 작거나 같으면 False

tf.cast
입력한 값의 결과를 지정한 자료형으로 변환해줌

tf.equal
tf.equal(x, y) : x, y를 비교하여 boolean 값을 반환
'''
accurcy = tf.reduce_mean(tf.cast(tf.equal(predicted,y), dtype=tf.float32))


with tf.Session() as sess: # Session을 close하지 않으려고
    sess.run(tf.global_variables_initializer())

    for step in range(10001):
        cost_val, _ = sess.run([cost, train], feed_dict={x: x_data, y: y_data})

        if step % 200 ==0:
            print(step, cost_val)
    h,c,a = sess.run([hypothesis, predicted, accurcy], feed_dict={x:x_data, y:y_data})

    print('\n Hypothesis : ', h ,'\n Correct (y) : ', c , '\n Accuracy : ', a)