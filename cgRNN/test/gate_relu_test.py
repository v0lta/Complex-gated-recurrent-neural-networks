import tensorflow as tf
import matplotlib.pyplot as plt


def gate_relu(z, reuse=None):
    with tf.variable_scope('gate_relu'):
        pre_act = tf.real(z)
        filter_neg = tf.nn.relu(pre_act)
        filter_pos = -tf.nn.relu(-filter_neg+1) + 1
    return filter_pos


test_graph = tf.Graph()
with test_graph.as_default():
    x = tf.linspace(-4.0, 4.0, 100)
    res = gate_relu(x)

with tf.Session(graph=test_graph):
    np_x = x.eval()
    np_res = res.eval()

plt.plot(np_x, np_res)
plt.show()
