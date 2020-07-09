import tensorflow as tf

node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0)
node3 = tf.add(node1, node2)

sses = tf.Session()
result = sses.run(node3)


print("node1 :", node1, "node2 :", node2)
print("node3 :", node3)

print(result)
# a = 3.0
# b = 4.0
# c = tf.Session.run(a*b)
# print(c)
print("sses.run(node1, node2) :", sses.run([node1, node2]))
print("sses.run(node3) :", sses.run(node3))