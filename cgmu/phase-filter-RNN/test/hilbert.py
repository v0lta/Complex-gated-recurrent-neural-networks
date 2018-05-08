import numpy as np
import tensorflow as tf
from IPython.core.debugger import Tracer
debug_here = Tracer()


#  ##  numpy hilbert transform ##
def np_hilbert(xr):
    n = xr.shape[0]
    # fft over columns.
    x = np.fft.fft(xr.transpose()).transpose()
    h = np.zeros([n])
    if n > 0 and 2*np.fix(n/2) == n:
        # even and nonempty
        h[0:int(n/2+1)] = 1
        h[1:int(n/2)] = 2
    elif n > 0:
        # odd and nonempty
        h[0] = 1
        h[1:int((n+1)/2)] = 2
    if len(x.shape) == 2:
        hs = np.stack([h]*x.shape[-1], -1)
    elif len(x.shape) == 1:
        hs = h
    else:
        raise NotImplementedError
    print(hs)
    return np.fft.ifft((x*hs).transpose()).transpose()


# Xr = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
# Xr = np.array([1.0, 2.0, 3.0, 4.0])
Xr = np.array([0.3, 0.0])
X = np_hilbert(Xr)
print(X)


#  # tensorflow hilbert transform ##

def hilbert(xr):
    with tf.variable_scope('hilbert_transform'):
        n = tf.Tensor.get_shape(xr).as_list()[0]
        # Run the fft on the columns no the rows.
        x = tf.transpose(tf.fft(tf.transpose(xr)))
        h = np.zeros([n])
        if n > 0 and 2*np.fix(n/2) == n:
            # even and nonempty
            h[0:int(n/2+1)] = 1
            h[1:int(n/2)] = 2
        elif n > 0:
            # odd and nonempty
            h[0] = 1
            h[1:int((n+1)/2)] = 2
        tf_h = tf.constant(h, name='h', dtype=tf.float32)
        if len(x.shape) == 2:
            hs = np.stack([h]*x.shape[-1], -1)
            reps = tf.Tensor.get_shape(x).as_list()[-1]
            hs = tf.stack([tf_h]*reps, -1)
        elif len(x.shape) == 1:
            hs = tf_h
        else:
            raise NotImplementedError
        tf_hc = tf.complex(hs, tf.zeros_like(hs))
        tmp = x*tf_hc
        return tf.transpose(tf.ifft(tf.transpose(tmp)))


test_graph = tf.Graph()
with test_graph.as_default():
    xr = tf.constant(Xr, dtype=tf.float32)
    xc = tf.complex(xr, tf.zeros_like(xr))
    X = hilbert(xc)

with tf.Session(graph=test_graph):
    print(X.eval())


