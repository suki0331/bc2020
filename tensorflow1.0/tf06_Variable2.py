# H = Wx+b

import tensorflow as tf

tf.set_random_seed(777)

W = tf.Variable(tf.random_normal([1]), name='Weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

x = [1,2,3]
W = tf.Variable([0.3], tf.float32)
b = tf.Variable([1], tf.float32)

sess = tf.Session()
sess.run(tf.compat.v1.global_variables_initializer())
aaa = sess.run(W)
# print("hypothesis: ", hypothesis)
print(aaa)
sess.close()

sess = tf.compat.v1.InteractiveSession()  # sess.run(W) 대신 W.eval()
sess.run(tf.compat.v1.global_variables_initializer())
bbb = W.eval()
# print("hypothesis: ", hypothesis)
print(bbb)
sess.close()

sess = tf.Session()
sess.run(tf.compat.v1.global_variables_initializer())
ccc = W.eval(session=sess)  # 그냥 session 도 eval 이 먹힘 하지만 session을 언급 이렇게 언급 해줘야됌
# print("hypothesis: ", hypothesis)
print(ccc)
sess.close()