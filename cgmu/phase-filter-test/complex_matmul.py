# complex matrix multiplication tests tensorflow.
import os
import numpy as np
import tensorflow as tf

from IPython.core.debugger import Tracer
debug_here = Tracer()

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

a = np.random.randn(100, 100)
b = np.random.randn(100, 100)

c = a + 1j*b

d = np.random.randn(100, 100)
e = np.random.randn(100, 100)
f = d + 1j*e

h = np.random.randn(100, 1) + 1j * np.random.randn(100, 1)
g = np.matmul(c, h)


def np_complex_matmul(c, h):
    cr = np.real(c)
    ci = np.imag(c)
    hr = np.real(h)
    hi = np.imag(h)

    A = np.concatenate([np.concatenate([cr, -ci], axis=1),
                        np.concatenate([ci, cr], axis=1)])
    x = np.concatenate([hr, hi])
    res = np.matmul(A, x)
    s = res.shape
    resr = res[:int(s[0]/2)]
    resi = res[int(s[0]/2):]
    return resr + 1j*resi


g_test = np_complex_matmul(c, h)
print(np.linalg.norm(g - g_test))


def tf_complex_matmul(c, h):
    cr = tf.real(c)
    ci = tf.imag(c)
    hr = tf.real(h)
    hi = tf.imag(h)

    A = tf.concat([tf.concat([cr, -ci], axis=1),
                   tf.concat([ci, cr], axis=1)], axis=0)
    x = tf.concat([hr, hi], axis=0)
    res = tf.matmul(A, x)
    s = tf.Tensor.get_shape(res).as_list()
    resr = res[:int(s[0]/2)]
    resi = res[int(s[0]/2):]
    return tf.complex(resr, resi)


test_graph = tf.Graph()
with test_graph.as_default():
    tf_a = tf.constant(a)
    tf_b = tf.constant(b)
    tf_c = tf.complex(a, b)

    tf_d = tf.constant(d)
    tf_e = tf.constant(e)
    tf_f = tf.complex(d, e)

    tf_h = tf.constant(h)
    tf_res = tf_complex_matmul(tf_c, tf_h)

sess = tf.Session(graph=test_graph)
with sess.as_default():
    g_tf = tf_res.eval()

print(np.linalg.norm(g - g_tf))
