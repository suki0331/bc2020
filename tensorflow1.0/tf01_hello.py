import tensorflow as tf
print(tf.__version__)

# 문자열 출력
hello = tf.constant("hello world")

print(hello)    #Tensor("Const:0", shape=(), dtype=string)
# print("hello world")
sess = tf.Session()
# 텐서에서 session을 통해 걸러서 나온다

# Tensorflow는 이러한 session 안에서만 실제적인 연산이나 로직을 수행하도록 되어 있습니다.
result = sess.run(hello) 
print(result)
# I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 
# b'hello world

'''
# tensorflow의 불편한 연산과정을 바로 출력하기 위해 keras가 나옴
# keras 의 백엔드엔 tensorflow가 들어가있다.
# tensorflow 2.0 부터 session을 없애버렸다.
# tensorflow 1점대 버전을 사용하면 session을 사용해야됌
'''

