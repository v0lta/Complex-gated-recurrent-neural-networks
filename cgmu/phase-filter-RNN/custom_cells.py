"""
    Implementation of the compelx memory cells of our NIPS-Paper.
    Including:
        1.) The original URNN-cell.
        2.) Our Phase-Relu cell.
"""
import collections
import numpy as np
import tensorflow as tf
from tensorflow import random_uniform_initializer as urnd_init
from IPython.core.debugger import Tracer
debug_here = Tracer()
_URNNStateTuple = collections.namedtuple("URNNStateTuple", ("o", "h"))


def unitary_init(shape, dtype=tf.float32, partition_info=None):
    limit = np.sqrt(6 / (shape[0] + shape[1]))
    rand_r = np.random.uniform(-limit, limit, shape[0:2])
    rand_i = np.random.uniform(-limit, limit, shape[0:2])
    crand = rand_r + 1j*rand_i
    debug_here()
    u, s, vh = np.linalg.svd(crand)
    # use u and vg to create a unitary matrix:
    debug_here()
    unitary = np.matmul(u, np.transpose(np.conj(vh)))

    test_eye = np.matmul(np.transpose(np.conj(unitary)), unitary)
    print('I - Wi.H Wi', np.linalg.norm(test_eye) - unitary)
    # test
    # plt.imshow(np.abs(np.matmul(unitary, np.transpose(np.conj(unitary))))); plt.show()
    stacked = np.stack([np.real(unitary), np.imag(unitary)], -1)
    assert stacked.shape == tuple(shape), "Unitary initialization shape mismatch."
    # debug_here()
    return tf.constant(stacked, dtype)


def arjovski_init(shape, dtype=tf.float32, partition_info=None):
    print("Arjosky basis initialization.")
    assert shape[0] == shape[1]
    omega1 = np.random.uniform(-np.pi, np.pi, shape[0])
    omega2 = np.random.uniform(-np.pi, np.pi, shape[0])
    omega3 = np.random.uniform(-np.pi, np.pi, shape[0])

    vr1 = np.random.uniform(-1, 1, [shape[0], 1])
    vi1 = np.random.uniform(-1, 1, [shape[0], 1])
    v1 = vr1 + 1j*vi1
    vr2 = np.random.uniform(-1, 1, [shape[0], 1])
    vi2 = np.random.uniform(-1, 1, [shape[0], 1])
    v2 = vr2 + 1j*vi2

    D1 = np.diag(np.exp(1j*omega1))
    D2 = np.diag(np.exp(1j*omega2))
    D3 = np.diag(np.exp(1j*omega3))

    vvh1 = np.matmul(v1, np.transpose(np.conj(v1)))
    beta1 = 2./np.matmul(np.transpose(np.conj(v1)), v1)
    R1 = np.eye(shape[0]) - beta1*vvh1

    vvh2 = np.matmul(v2, np.transpose(np.conj(v2)))
    beta2 = 2./np.matmul(np.transpose(np.conj(v2)), v2)
    R2 = np.eye(shape[0]) - beta2*vvh2

    perm = np.random.permutation(np.eye(shape[0], dtype=np.float32)) \
        + 1j*np.zeros(shape[0])

    fft = np.fft.fft
    ifft = np.fft.ifft

    step1 = fft(D1)
    step2 = np.matmul(R1, step1)
    step3 = np.matmul(perm, step2)
    step4 = np.matmul(D2, step3)
    step5 = ifft(step4)
    step6 = np.matmul(R2, step5)
    unitary = np.matmul(D3, step6)
    eye_test = np.matmul(np.transpose(np.conj(unitary)), unitary)
    unitary_test = np.linalg.norm(np.eye(shape[0]) - eye_test)
    print('I - Wi.H Wi', unitary_test, unitary.dtype)
    assert unitary_test < 1e-10, "Unitary initialization not unitary enough."
    stacked = np.stack([np.real(unitary), np.imag(unitary)], -1)
    assert stacked.shape == tuple(shape), "Unitary initialization shape mismatch."
    return tf.constant(stacked, dtype)


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


def mod_relu(z, scope='', reuse=None):
    """
        Implementation of the modRelu from Arjovski et al.
        f(z) = relu(|z| + b)(z / |z|) or
        f(r,theta) = relu(r + b)e^(i*theta)
        b is initialized to zero, this leads to a network, which
        is linear during early optimization.
    Input:
        z: complex input.
        b: 'dead' zone radius.
    Returns:
        z_out: complex output.
    """
    with tf.variable_scope('mod_relu' + scope, reuse=reuse):
        b = tf.get_variable('b', [], dtype=tf.float32,
                            initializer=urnd_init(-0.01, 0.01))
        modulus = tf.sqrt(tf.real(z)**2 + tf.imag(z)**2)
        rescale = tf.nn.relu(modulus + b) / (modulus + 1e-6)
        # return tf.complex(rescale * tf.real(z),
        #                   rescale * tf.imag(z))
        rescale = tf.complex(rescale, tf.zeros_like(rescale))
        return tf.multiply(rescale, z)


def phase_relu(z, scope='', reuse=None, coupled=False):
    """
        Set up the Phase Relu non-linearity from our paper.
    """
    def richards(n, k):
        """ Elementwise implementation of the richards-function step.
            Has non-zero gradient. Not holomorph, but trainable.
        """
        with tf.variable_scope("richards"):
            return tf.nn.sigmoid(k*n)

    def step(n):
        return tf.cast(n > 0, tf.float32)  # this is bad grad=0!

    with tf.variable_scope('phase_relu' + scope, reuse=reuse):
        a = tf.get_variable('a', [], dtype=tf.float32,
                            initializer=urnd_init(0.49, 0.51))
        b = tf.get_variable('b', [], dtype=tf.float32,
                            initializer=urnd_init(0.49, 0.51))
        k = tf.constant(10.0)
        pi = tf.constant(np.pi)
        if coupled:
            a = -a
            b = -b
        theta = tf.atan2(tf.real(z), tf.imag(z))
        # g = step(tf.sin(theta*0.6 + 0.5*pi))
        g = richards(tf.sin(theta*a + b*pi), k)
        g = tf.complex(tf.ones_like(g), tf.zeros_like(g))
        return tf.multiply(g, z)


def hirose(z, scope='', reuse=None):
    """
    Compute the non-linearity proposed by Hirose.
    """
    with tf.variable_scope('hirose' + scope, reuse=reuse):
        m = tf.get_variable('m', [], tf.float32,
                            initializer=urnd_init(0.9, 1.1))
        modulus = tf.sqrt(tf.real(z)**2 + tf.imag(z)**2)
        rescale = tf.complex(tf.nn.tanh(modulus/m)/modulus,
                             tf.zeros_like(modulus))
        return tf.multiply(rescale, z)


def moebius(z, scope='', reuse=None):
    """
    Implement a learnable moebius transformation.
    """
    with tf.variable_scope('moebius' + scope, reuse=reuse):
        ar = tf.get_variable('ar', [], tf.float32,
                             initializer=tf.constant_initializer(1))
        ai = tf.get_variable('ai', [], tf.float32,
                             initializer=tf.constant_initializer(0))
        b = tf.get_variable('b', [2], tf.float32,
                            initializer=tf.constant_initializer(0))
        c = tf.get_variable('c', [2], tf.float32,
                            initializer=tf.constant_initializer(0))
        dr = tf.get_variable('dr', [], tf.float32,
                             initializer=tf.constant_initializer(1))
        di = tf.get_variable('di', [], tf.float32,
                             initializer=tf.constant_initializer(0))

        a = tf.complex(ar, ai)
        b = tf.complex(b[0], b[1])
        c = tf.complex(c[0], c[1])
        d = tf.complex(dr, di)
        return tf.divide(tf.multiply(a, z) + b,
                         tf.multiply(c, z) + d)


def modSigmoid(z, reuse=None):
    """
    ModSigmoid implementation.
    """
    with tf.variable_scope('modSigmoid'):
        alpha = tf.get_variable('alpha', [], dtype=tf.float32)
        pre_act = alpha * tf.real(z) + (1 - alpha)*tf.imag(z)
        return tf.nn.sigmoid(pre_act)


def linear(z, scope='', reuse=None, coupled=False):
    return z


def rfl_mul(h, state_size, no, reuse):
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
        vr = tf.get_variable('vr', shape=[state_size, 1], dtype=tf.float32,
                             initializer=tf.glorot_uniform_initializer())
        vi = tf.get_variable('vi', shape=[state_size, 1], dtype=tf.float32,
                             initializer=tf.glorot_uniform_initializer())

    with tf.variable_scope("ref_mul_" + str(no), reuse=reuse):
        hr = tf.real(h)
        hi = tf.imag(h)
        vstarv = tf.reduce_sum(vr**2 + vi**2)
        hr_vr = tf.matmul(hr, vr)
        hr_vi = tf.matmul(hr, vi)
        hi_vr = tf.matmul(hi, vr)
        hi_vi = tf.matmul(hi, vi)

        # tf.matmul with transposition is the same as T.outer
        # we need something of the shape [batch_size, state_size] in the end
        a = tf.matmul(hr_vr - hi_vi, vr, transpose_b=True)
        b = tf.matmul(hr_vi + hi_vr, vi, transpose_b=True)
        c = tf.matmul(hr_vr - hi_vi, vi, transpose_b=True)
        d = tf.matmul(hr_vi + hi_vr, vr, transpose_b=True)

        # the thing we return is:
        # return_re = hr - (2/vstarv)(d - c)
        # return_im = hi - (2/vstarv)(a + b)
        new_hr = hr - (2.0 / vstarv) * (a + b)
        new_hi = hi - (2.0 / vstarv) * (d - c)
        new_state = tf.complex(new_hr, new_hi)

        # v = tf.complex(vr, vi)
        # vstarv = tf.complex(tf.reduce_sum(vr**2 + vi**2), 0.0)
        # # vstarv = tf.matmul(tf.transpose(tf.conj(v)), v)
        # vvstar = tf.matmul(v, tf.transpose(tf.conj(v)))
        # refsub = (2.0/vstarv)*vvstar
        # R = tf.identity(refsub) - refsub
        return new_state


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
        omega = tf.get_variable('vr', shape=[state_size], dtype=tf.float32,
                                initializer=urnd_init(-np.pi, np.pi))
        dr = tf.cos(omega)
        di = tf.sin(omega)

    with tf.variable_scope("diag_mul_" + str(no)):
        D = tf.diag(tf.complex(dr, di))
        return tf.matmul(h, D)


def permutation(h, state_size, no, reuse):
    """
    Apply a random permutation to the RNN state.
    Input:
        h: the original state.
    Output:
        hp: the permuted state.
    """
    with tf.variable_scope("permutation_" + str(no), reuse):
        def _initializer(shape, dtype=np.float32, partition_info=None):
            return np.random.permutation(np.eye(state_size, dtype=np.float32))
        Pr = tf.get_variable("Permutation", dtype=tf.float32,
                             initializer=_initializer, shape=[state_size],
                             trainable=False)
        P = tf.complex(Pr, tf.constant(0.0, dtype=tf.float32))
    return tf.matmul(h, P)


def matmul_plus_bias(x, num_proj, scope, reuse, bias_init=0.0, orthogonal=False):
    """
    Compute Ax + b.
    Input: x
    Returns: Ax + b
    """
    in_shape = tf.Tensor.get_shape(x).as_list()
    with tf.variable_scope(scope, reuse=reuse):
        if orthogonal:
            with tf.variable_scope('orthogonal_stiefel', reuse=reuse):
                A = tf.get_variable('gate_O', [in_shape[-1], num_proj],
                                    dtype=tf.float32,
                                    initializer=tf.orthogonal_initializer())
        else:
            A = tf.get_variable('A', [in_shape[-1], num_proj], dtype=tf.float32,
                                initializer=tf.glorot_uniform_initializer())
        b = tf.get_variable('bias', [num_proj], dtype=tf.float32,
                            initializer=tf.constant_initializer(bias_init))
        print('Initializing', tf.contrib.framework.get_name_scope(), 'bias to',
              bias_init)
    with tf.variable_scope('linear_layer'):
        return tf.matmul(x, A) + b


def complex_matmul_plus_bias(x, num_proj, scope, reuse, bias_init=0.0,
                             unitary=False, orthogonal=False,
                             unitary_init=arjovski_init):
    """
    Compute Ax + b.
    Input: x
    Returns: Ax + b
    """
    in_shape = tf.Tensor.get_shape(x).as_list()
    # debug_here()
    with tf.variable_scope(scope, reuse=reuse):
        if unitary:
            with tf.variable_scope('unitary_stiefel', reuse=reuse):
                varU = tf.get_variable('gate_U',
                                       shape=in_shape[-1:] + [num_proj] + [2],
                                       dtype=tf.float32,
                                       initializer=unitary_init)
                A = tf.complex(varU[:, :, 0], varU[:, :, 1])
        elif orthogonal:
            with tf.variable_scope('orthogonal_stiefel', reuse=reuse):
                Ar = tf.get_variable('gate_Ur', in_shape[-1:] + [num_proj],
                                     dtype=tf.float32,
                                     initializer=tf.orthogonal_initializer())
                Ai = tf.get_variable('gate_Ui', in_shape[-1:] + [num_proj],
                                     dtype=tf.float32,
                                     initializer=tf.orthogonal_initializer())
                A = tf.complex(Ar, Ai)
        else:
            varU = tf.get_variable('gate_A',
                                   shape=in_shape[-1:] + [num_proj] + [2],
                                   dtype=tf.float32,
                                   initializer=tf.glorot_uniform_initializer())
            A = tf.complex(varU[:, :, 0], varU[:, :, 1])

        varb = tf.get_variable('bias_g', [num_proj] + [2], dtype=tf.float32,
                               initializer=tf.constant_initializer(bias_init))
        b = tf.complex(varb[:, 0], varb[:, 1])
        return tf.matmul(x, A) + b


def C_to_R(h, num_proj, reuse, scope=None, bias_init=0.0):
    with tf.variable_scope(scope or "C_to_R"):
        concat = tf.concat([tf.real(h), tf.imag(h)], axis=-1)
        return matmul_plus_bias(concat, num_proj, 'final', reuse, bias_init)


class UnitaryCell(tf.nn.rnn_cell.RNNCell):
    """
    Tensorflow implementation of unitary evolution RNN as proposed by Arjosky et al.
    """
    def __init__(self, num_units, activation=mod_relu, num_proj=None, reuse=None):
        super().__init__(_reuse=reuse)
        self._num_units = num_units
        self._activation = activation
        self._output_size = num_proj
        self._arjovski_basis = False

    def to_string(self):
        cell_str = 'UnitaryCell' + '_' \
            + '_' + 'activation' + '_' + str(self._activation.__name__) + '_' \
            + '_arjovski_basis' + '_' + str(self._arjovski_basis)
        return cell_str

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
        out = tf.zeros([batch_size, self._output_size], dtype=tf.float32)
        # first_state = tf.complex(tf.zeros([batch_size, self._num_units],
        #                                   dtype=tf.float32),
        #                          tf.zeros([batch_size, self._num_units],
        #                                   dtype=tf.float32))
        # bucket = np.sqrt(3.0/self._num_units)
        # # TODO: Test this!!!
        # first_state = tf.complex(tf.random_uniform([batch_size, self._num_units],
        #                          minval=-bucket, maxval=bucket, dtype=dtype),
        #                          tf.random_uniform([batch_size, self._num_units],
        #                          minval=-bucket, maxval=bucket, dtype=dtype))
        omegas = tf.random_uniform([batch_size, self._num_units],
                                   minval=0, maxval=2*np.pi)
        sx = tf.cos(omegas)
        sy = tf.sin(omegas)
        r = (1.0)/np.sqrt(self._num_units)
        first_state = tf.complex(r*sx, r*sy)
        return URNNStateTuple(out, first_state)

    # TODO h0.

    def call(self, inputs, state):
        """
            Evaluate the RNN cell. Using
            h_(t+1) = U_t*f(h_t) + V_t x_t
        """
        with tf.variable_scope("UnitaryCell"):
            last_out, last_h = state
            if self._arjovski_basis:
                with tf.variable_scope("arjovski_basis", reuse=self._reuse):
                    step1 = diag_mul(last_h, self._num_units, 0, self._reuse)
                    step2 = tf.spectral.fft(step1)
                    step3 = rfl_mul(step2, self._num_units, 0, self._reuse)
                    step4 = permutation(step3, self._num_units, 0, self._reuse)
                    step5 = diag_mul(step4, self._num_units, 1, self._reuse)
                    step6 = tf.spectral.ifft(step5)
                    step7 = rfl_mul(step6, self._num_units, 1, self._reuse)
                    Uh = diag_mul(step7, self._num_units, 2, self._reuse)
            else:
                with tf.variable_scope("unitary_stiefel", reuse=self._reuse):
                    varU = tf.get_variable("recurrent_U",
                                           shape=[self._num_units, self._num_units, 2],
                                           dtype=tf.float32,
                                           initializer=arjovski_init)
                    U = tf.complex(varU[:, :, 0], varU[:, :, 1])
                    # U = tf.Print(U, [U])
                Uh = tf.matmul(last_h, U)

            # Deal with the inputs
            # Mapping inputs into the complex plane, by folding:
            Vxr = matmul_plus_bias(inputs, self._num_units, 'real', self._reuse)
            Vxi = matmul_plus_bias(inputs, self._num_units, 'imag', self._reuse)
            Vx = tf.complex(Vxr, Vxi)
            # By leaving the real part intact.
            # Vx = tf.complex(Vxr, tf.tf.zeros_like(Vxr))

            # By FFT.
            # TODO.
            zt = Uh + Vx
            ht = self._activation(zt, '', self._reuse)

            # Mapping the state back onto the real axis.
            # By mapping.

            output = C_to_R(ht, self._output_size, reuse=self._reuse)

            # By fft.
            # TODO.
            # debug_here()
            # print('dbg')
            newstate = URNNStateTuple(output, ht)
        return output, newstate


class UnitaryMemoryCell(UnitaryCell):
    """
    Tensorflow implementation of unitary evolution RNN as proposed by Arjosky et al.
    """

    def __init__(self, num_units, activation=moebius, num_proj=None, reuse=None,
                 orthogonal_gate=False, unitary_gate=False):
        super().__init__(num_units, num_proj=num_proj, reuse=reuse, )
        self._activation = activation  # FIXME: beat linear.
        self._output_activation = None  # TODO.
        self._single_gate = True
        self._orthogonal_gate = orthogonal_gate
        self._unitary_gate = unitary_gate
        self._arjovski_basis = False

        if orthogonal_gate and unitary_gate:
            raise ValueError("Gates cannot be split orthogonal and unitary.")

    def to_string(self):
        cell_str = 'UnitaryMemoryCell' + '_' \
            + '_activation' + '_' + str(self._activation.__name__) \
            + '_singleGate' + '_' + str(self._single_gate) + '_' \
            + '_orthogonalGate' + '_' + str(self._orthogonal_gate) \
            + '_unitaryGate' + '_' + str(self._unitary_gate) \
            + '_arjovskiBasis' + '_' + str(self._arjovski_basis)
        return cell_str

    def complex_memory_gate(self, h, x, scope, reuse, bias_init=0.0):
        """
        Produce a bounded gate output mapping from C to
        R. This operation breaks complex gradients, but
        Wirtinger Caclulus may be used to justify the
        gradients used here as approximately correct.
        """
        with tf.variable_scope(scope, reuse):
            hr = C_to_R(h, self._num_units, reuse, scope='C_to_R_h', bias_init=bias_init)
            xr = C_to_R(x, self._num_units, reuse, scope='C_to_R_x', bias_init=bias_init)
            scale = tf.nn.sigmoid(hr + xr)
            return tf.complex(scale, tf.zeros_like(scale))

    def single_memory_gate(self, h, x, scope, reuse, bias_init=0.0,
                           unitary=False, orthogonal=False):
        """
        New unified gate.
        """
        with tf.variable_scope(scope, reuse):
            gh = complex_matmul_plus_bias(h, self._num_units,
                                          scope='gh',
                                          reuse=reuse, bias_init=bias_init,
                                          unitary=unitary,
                                          orthogonal=orthogonal)
            gx = complex_matmul_plus_bias(x, self._num_units,
                                          scope='gx', reuse=reuse, bias_init=bias_init)
            g = gh + gx
            ig = tf.nn.sigmoid(tf.real(g))
            fg = tf.nn.sigmoid(tf.imag(g))
            return (tf.complex(ig, tf.zeros_like(ig), name='ig'),
                    tf.complex(fg, tf.zeros_like(fg), name='fg'))

    def call(self, inputs, state):
        """
            Evaluate the RNN cell. Using
            h_(t+1) = U_t*f(h_t) + V_t x_t
        """
        with tf.variable_scope("UnitaryMemoryCell"):
            last_out, last_h = state
            if self._arjovski_basis:
                with tf.variable_scope("arjovski_basis", reuse=self._reuse):
                    step1 = diag_mul(last_h, self._num_units, 0, self._reuse)
                    step2 = tf.spectral.fft(step1)
                    step3 = rfl_mul(step2, self._num_units, 0, self._reuse)
                    step4 = permutation(step3, self._num_units, 0, self._reuse)
                    step5 = diag_mul(step4, self._num_units, 1, self._reuse)
                    step6 = tf.spectral.ifft(step5)
                    step7 = rfl_mul(step6, self._num_units, 1, self._reuse)
                    Uh = diag_mul(step7, self._num_units, 2, self._reuse)
            else:
                with tf.variable_scope("unitary_stiefel", reuse=self._reuse):
                    varU = tf.get_variable("recurrent_U",
                                           shape=[self._num_units, self._num_units, 2],
                                           dtype=tf.float32,
                                           initializer=arjovski_init)
                    U = tf.complex(varU[:, :, 0], varU[:, :, 1])
                Uh = tf.matmul(last_h, U)

            # Deal with the inputs
            # Mapping inputs into the complex plane, by folding:
            if not self._single_gate:
                Vxr = matmul_plus_bias(inputs, self._num_units,
                                       scope='real', reuse=self._reuse,
                                       orthogonal=self._orthogonal_gate)
                Vxi = matmul_plus_bias(inputs, self._num_units,
                                       scope='imag', reuse=self._reuse,
                                       orthogonal=self._orthogonal_gate)
                Vx = tf.complex(Vxr, Vxi)
            else:
                x = tf.complex(inputs, tf.zeros_like(inputs))
                Vx = complex_matmul_plus_bias(x, self._num_units,
                                              scope='input_weights',
                                              reuse=self._reuse,
                                              unitary=True,
                                              unitary_init=unitary_init)
            # By leaving the real part intact.
            # Vx = tf.complex(Vxr, tf.tf.zeros_like(Vxr))
            # By FFT.
            # TODO.
            # ################# Hilbert transform. ####################
            # TODO.

            if not self._single_gate:
                ig = self.complex_memory_gate(Uh, Vx, scope='input_gate',
                                              reuse=self._reuse, bias_init=1.0)
                fg = self.complex_memory_gate(Uh, Vx, scope='forget_gate',
                                              reuse=self._reuse, bias_init=1.0)
            else:
                ig, fg = self.single_memory_gate(Uh, Vx,
                                                 scope='single_gate',
                                                 reuse=self._reuse,
                                                 bias_init=1.0,
                                                 unitary=self._unitary_gate,
                                                 orthogonal=self._orthogonal_gate
                                                 )
            pre_h = tf.multiply(fg, Uh) + tf.multiply(ig, Vx)
            ht = self._activation(pre_h, reuse=self._reuse)

            # Mapping the state back onto the real axis.
            # By mapping.
            output = C_to_R(ht, self._output_size, reuse=self._reuse)
            # ht = self._activation(ht, '', reuse=True, coupled=True)

            # By ifft.
            # TODO.
            # debug_here()
            # print('dbg')
            newstate = URNNStateTuple(output, ht)
        return output, newstate


