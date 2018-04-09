"""
    Implementation of the compelx memory cells of our NIPS-Paper.
    Including:
        1.) The original URNN-cell.
        2.) Our Phase-Relu cell.
"""
import collections
import numpy as np
import tensorflow as tf
import ipdb; ipdb.set_trace()

### !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# TODO: INITIALIZATION !!
### !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

_URNNStateTuple = collections.namedtuple("URNNStateTuple", ("o", "h"))


class URNNStateTuple(_URNNStateTuple):
    """Tuple used by LSTM Cells for `state_size`, `zero_state`, and output state.
       Stores two elements: `(c, h)`, in that order.
       Only used when `state_is_tuple=True`.
    """
    slots__ = ()

    @property
    def dtype(self):
        (c, h) = self
        if c.dtype != h.dtype:
            raise TypeError("Inconsistent internal state: %s vs %s" %
                            (str(c.dtype), str(h.dtype)))
        return c.dtype


def modRelu(z, reuse):
    """
        Implementation of the modRelu from Arjovski et al.
        f(z) = relu(|z| + b)(z / |z|) or
        f(r,theta) = relu(r + b)e^(i*theta)
    Input:
        z: complex input.
        b: 'dead' zone radius.
    Returns:
        z_out: complex output.
    """
    with tf.variable_scope('modRelu', reuse=reuse):
        b = tf.get_variable('b', [1, 1], dtype=tf.float32)
        modulus = tf.sqrt(tf.real(z)**2 + tf.imag(z)**2)
        return tf.nn.relu(modulus + b) * (z/modulus)


def PhaseRelu(z, a, b):
    """
        Set up the Phase Relu non-linearity from our paper.
    """
    pass


def ref_mul(h, state_size, no, reuse):
    """
    Multiplication with a reflection.
    Implementing R = I - (vv*/|v|^2)
    Input:
        h: hidden state_vector.
        state_size: The RNN state size.
        reuse: True if graph variables should be reused.
    Returns:
        R*h
    """
    with tf.variable_scope("reflection_v_" + str(no), reuse=reuse):
        vr = tf.get_variable('vr', shape=[state_size, 1], dtype=tf.float32)
        vi = tf.get_variable('vi', shape=[state_size, 1], dtype=tf.float32)

    with tf.variable_scope("ref_mul:" + str(no), reuse=reuse):
        v = tf.complex(vr, vi)
        vstarv = tf.complex(tf.reduce_sum(vr**2, vi**2), 0)
        refsub = (2/vstarv)*tf.matmul(v, tf.transpose(tf.conj(v)))
        R = tf.identity(refsub) - refsub
        return tf.matmul(R, h)


def diag_mul(h, state_size, no, reuse):
    """
    Multiplication with a diagonal matrix.
    Input:
        h: hidden state_vector.
        state_size: The RNN state size.
        reuse: True if graph variables should be reused.
    Returns:
        R*h
    """

    with tf.variable_scope("diag_phis_" + str(no), reuse=reuse):
        # TODO: Enforce lambda = 1!!
        omega = tf.get_variable('vr', shape=[state_size, 1], dtype=tf.float32)
        dr = tf.cos(omega)
        di = tf.sin(omega)

    with tf.variable_scope("diag_mul_" + str(no)):
        D = tf.diag(tf.complex(dr, di))
        return tf.matmul(D, h)


def permutation(h, state_size, no, reuse):
    """
    Apply a random permutation to the RNN state.
    Input:
        h: the original state.
    Output:
        hp: the permuted state.
    """
    with tf.variable_scope("permutation_" + str(no), reuse):
        init = tf.complex(np.random.permutation(np.eye(state_size, dtype=np.float32)),
                          tf.constant(0.0, dtype=tf.float32))
        P = tf.get_variable("Permutation", dtype=tf.complex64,
                            initializer=init,
                            trainable=False)
    return tf.matmul(P, h)


def matmul_plus_bias(x, output_size, reuse):
    """
    Compute Ax + b.
    Input: x
    Returns: Ax + b
    """
    in_shape = tf.Tensor.get_shape(x).as_list()
    with tf.variable_scope("linear", reuse=reuse):
        A = tf.get_variable('A', [in_shape[-1:], output_size], dtype=tf.float32)
        b = tf.get_variable('b', [output_size, 1], dtype=tf.float32)
    with tf.variable_scope('linar_layer'):
        return tf.matmul(A, x) + b


def complex_matmul_plus_bias(x, output_size, reuse):
    """
    Compute Ax + b.
    Input: x
    Returns: Ax + b
    """
    in_shape = tf.Tensor.get_shape(x).as_list()
    with tf.variable_scope("linear", reuse=reuse):
        Ar = tf.get_variable('Ar', [in_shape[-1:], output_size], dtype=tf.float32)
        Ai = tf.get_variable('Ai', [in_shape[-1:], output_size], dtype=tf.float32)
        br = tf.get_variable('br', [output_size, 1], dtype=tf.float32)
        bi = tf.get_variable('bi', [output_size, 1], dtype=tf.float32)
        A = tf.complex(Ar, Ai)
        b = tf.complex(br, bi)
    with tf.variable_scope('complex_linar_layer'):
        return tf.matmul(A, x) + b


def C_to_R(h, output_size, reuse):
    with tf.variable_scope("C_to_R"):
        concat = tf.concatinate([tf.real(h). tf.imag(h)], axis=-1)
        return matmul_plus_bias(concat, output_size, reuse)


class UnitaryCell(tf.nn.rnn_cell.RNNCell):
    """
    Tensorflow implementation of unitary evolution RNN as proposed by Arjosky et al.
    """

    def __init__(self, num_units, output_size=None, reuse=None):
        super().__init__(_reuse=reuse)
        self._num_units = num_units
        self._activation = modRelu
        self._output_size = output_size

    @property
    def state_size(self):
        return URNNStateTuple(self._num_units, self._num_units)

    @property
    def output_size(self):
        if self._output_size is None:
            return self._num_units
        else:
            return self._output_size

    def zero_state(self, batch_size, dtype=tf.float32):
        out = tf.zeros([self._output_size, 1], dtype=tf.float32)
        hidden = tf.complex(tf.zeros([self._num_units, 1], dtype=tf.float32),
                            tf.zeros([self._num_units, 1], dtype=tf.float32))
        return URNNStateTuple(out, hidden)

    def call(self, inputs, state, reuse=False):
        """
            Evaluate the RNN cell. Using
            h_(t+1) = U_t*f(h_t) + V_t x_t
        """
        with tf.variable_scope("UnitaryCell"):
            last_out, last_h = state
            # Compute the hidden part.
            step1 = diag_mul(last_h, self._num_units, 0, self._reuse)
            step2 = tf.spectral.fft(step1)
            step3 = tf.ref_mul(step2, self._num_units, 0, self._reuse)
            step4 = permutation(step3, self.num_units, 0, self._reuse)
            step5 = diag_mul(step4, self._num_units, 1, self._reuse)
            step6 = tf.spectral.ifft(step5, self.num_units)
            step7 = tf.ref_mul(step6, self.num_units, 1, self._reuse)
            Uh = diag_mul(step7, self.num_units, 2, self._reuse)

            # Deal with the inputs
            # Mapping inputs into the complex plane, by folding:
            Vxr = matmul_plus_bias(inputs, self._num_units, self._reuse)
            Vxi = matmul_plus_bias(inputs, self._num_units, self._reuse)
            Vx = tf.complex(Vxr, Vxi)
            # By leaving the real part intact.
            # Vx = tf.complex(Vxr, tf.tf.zeros_like(Vxr))

            # By FFT.
            # TODO.

            zt = Uh + Vx
            ht = self._activation(zt)

            # Mapping the state back onto the real axis.
            # By mapping.
            output = C_to_R(ht, self.output_size)

            # By fft.
            # TODO.

        return URNNStateTuple(output, ht)
