# 3 + 4 + 5 
# 4 - 3
# 3 * 4
# 4 / 2

# 2 3 4 5

import tensorflow as tf

a = tf.constant(2)
b = tf.constant(3)
c = tf.constant(4)
d = tf.constant(5)

add = tf.add(b, c)
add = tf.add(add, d)

sub = tf.subtract(c, b)

mul = tf.multiply(b, c)

div = tf.divide(c, a)

sses = tf.Session()

print("add:", sses.run(add))
print("sub:", sses.run(sub))
print("mul:", sses.run(mul))
print("div:", sses.run(div))
