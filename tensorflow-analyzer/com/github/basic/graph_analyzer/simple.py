import tensorflow as tf
from tensorflow.python.client import device_lib
import os

tf.compat.v1.disable_eager_execution()

print(tf.test.is_gpu_available())
print(device_lib.list_local_devices())

g1 = tf.Graph()
with g1.as_default():
    a = tf.constant([[1,2,3],[4,5,6]],dtype=tf.float32)
    b = tf.Variable(tf.random.normal([5,2],seed=12.34,name="b"))
    m1 = tf.matmul(b,a,name="matmul")
    m2 = tf.nn.relu(m1,name="relu")

# with tf.device('/device:CPU:0'):


print(tf.Graph.get_operations(g1).__len__())
# for temp in tf.Graph.get_operations(g1):
#      print(temp)

tf.compat.v1.get_default_session().run(a)
sess.run(a)
sess.run(b)