
""" Extensions to TF RNN class by una_dinosaria"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import numpy as np
import tensorflow as tf
from IPython.core.debugger import Tracer
debug_here = Tracer()

from tensorflow.contrib.rnn.python.ops.core_rnn_cell import RNNCell

# The import for LSTMStateTuple changes in TF >= 1.2.0
from pkg_resources import parse_version as pv
if pv(tf.__version__) >= pv('1.2.0'):
  from tensorflow.contrib.rnn import LSTMStateTuple
else:
  from tensorflow.contrib.rnn.python.ops.core_rnn_cell import LSTMStateTuple
del pv

from tensorflow.python.ops import variable_scope as vs
from tensorflow import random_uniform_initializer as urnd_init

import collections
import math

class ResidualWrapper(RNNCell):
  """Operator adding residual connections to a given cell."""

  def __init__(self, cell):
    """Create a cell with added residual connection.

    Args:
      cell: an RNNCell. The input is added to the output.

    Raises:
      TypeError: if cell is not an RNNCell.
    """
    if not isinstance(cell, RNNCell):
      raise TypeError("The parameter cell is not a RNNCell.")

    self._cell = cell

  @property
  def state_size(self):
    return self._cell.state_size

  @property
  def output_size(self):
    return self._cell.output_size

  def __call__(self, inputs, state, scope=None):
    """Run the cell and add a residual connection."""

    # Run the rnn as usual
    output, new_state = self._cell(inputs, state, scope)

    # Add the residual connection
    output = tf.add(output, inputs)

    return output, new_state

class LinearSpaceDecoderWrapper(RNNCell):
  """Operator adding a linear encoder to an RNN cell"""

  def __init__(self, cell, output_size):
    """Create a cell with with a linear encoder in space.

    Args:
      cell: an RNNCell. The input is passed through a linear layer.

    Raises:
      TypeError: if cell is not an RNNCell.
    """
    if not isinstance(cell, RNNCell):
      raise TypeError("The parameter cell is not a RNNCell.")

    self._cell = cell

    print( 'output_size = {0}'.format(output_size) )
    print( ' state_size = {0}'.format(self._cell.state_size) )

    # Tuple if multi-rnn
    if isinstance(self._cell.state_size,tuple):

      # Fine if GRU...
      insize = self._cell.state_size[-1]

      # LSTMStateTuple if LSTM
      if isinstance( insize, LSTMStateTuple ):
        insize = insize.h
    else:
      # Fine if not multi-rnn
      insize = self._cell.state_size

    self.w_out = tf.get_variable("proj_w_out",
        [insize, output_size],
        dtype=tf.float32,
        initializer=tf.random_uniform_initializer(minval=-0.04, maxval=0.04))
    self.b_out = tf.get_variable("proj_b_out", [output_size],
        dtype=tf.float32,
        initializer=tf.random_uniform_initializer(minval=-0.04, maxval=0.04))

    self.linear_output_size = output_size

  @property
  def state_size(self):
    return self._cell.state_size

  @property
  def output_size(self):
    return self.linear_output_size

  def __call__(self, inputs, state, scope=None):
    """Use a linear layer and pass the output to the cell."""

    # Run the rnn as usual
    output, new_state = self._cell(inputs, state, scope)

    # Apply the multiplication to everything
    output = tf.matmul(output, self.w_out) + self.b_out

    return output, new_state


def mod_sigmoid_beta(z, scope='', reuse=None):
    """
    ModSigmoid implementation, with uncoupled alpha and beta.
    """
    with tf.variable_scope('mod_sigmoid_beta_' + scope, reuse=reuse):
        alpha = tf.get_variable('alpha', [], dtype=tf.float32,
                                initializer=tf.constant_initializer(0.0))
        beta = tf.get_variable('beta', [], dtype=tf.float32,
                               initializer=tf.constant_initializer(1.0))
        alpha_norm = tf.nn.sigmoid(alpha)
        beta_norm = tf.nn.sigmoid(beta)
        pre_act = alpha_norm * tf.real(z) + beta_norm*tf.imag(z)
        return tf.complex(tf.nn.sigmoid(pre_act), tf.zeros_like(pre_act))


def mod_sigmoid_prod(z, scope='', reuse=None):
    """
    ModSigmoid implementation.
    """
    with tf.variable_scope('mod_sigmoid_prod_' + scope, reuse=reuse):
        prod = tf.nn.sigmoid(tf.real(z)) * tf.nn.sigmoid(tf.imag(z))
        return tf.complex(prod, tf.zeros_like(prod))


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

def complex_matmul(x, num_proj, scope, reuse, bias=False, bias_init_r=0.0,
                   bias_init_c=0.0, unitary=False, orthogonal=False,
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
        if bias:
            varbr = tf.get_variable('bias_r', [num_proj], dtype=tf.float32,
                                    initializer=tf.constant_initializer(bias_init_r))
            varbc = tf.get_variable('bias_c', [num_proj], dtype=tf.float32,
                                    initializer=tf.constant_initializer(bias_init_c))
            b = tf.complex(varbr, varbc)
            return tf.matmul(x, A) + b
        else:
            return tf.matmul(x, A)

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


def C_to_R(h, num_proj, reuse, scope=None, bias_init=0.0):
    with tf.variable_scope(scope or "C_to_R"):
        concat = tf.concat([tf.real(h), tf.imag(h)], axis=-1)
        return matmul_plus_bias(concat, num_proj, 'final', reuse, bias_init)



_URNNStateTuple = collections.namedtuple("URNNStateTuple", ("o", "h"))

class URNNStateTuple(_URNNStateTuple):
    """Tuple used by URNN Cells for `state_size`, `zero_state`, and output state.
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

class ComplexGatedRecurrentUnit(RNNCell):
    '''
    Can we implement a complex GRU?
    '''
    def __init__(self, num_units, activation=mod_relu,
                 num_proj=None, reuse=None, single_gate=True,
                 complex_out=False):
        super().__init__(_reuse=reuse)
        self._num_units = num_units
        self._activation = activation
        # self._state_to_state_act = linear
        self._num_proj = num_proj
        self._arjovski_basis = False
        self._input_fourier = False
        self._input_hilbert = False
        self._input_split_matmul = False
        self._stateU = True
        self._gateO = False
        self._single_gate = single_gate
        self._gate_activation = mod_sigmoid_beta
        self._single_gate_avg = False
        self._complex_inout = complex_out

    def to_string(self):
        cell_str = 'ComplexGatedRecurrentUnit' + '_' \
            + '_' + 'activation' + '_' + str(self._activation.__name__) + '_'
        if self._input_fourier:
            cell_str += '_input_fourier_'
        elif self._input_hilbert:
            cell_str += '_input_hilbert_'
        elif self._input_split_matmul:
            cell_str += '__input_split_matmul_'
        cell_str += '_stateU' + '_' + str(self._stateU) \
                    + '_gateO_' + str(self._gateO) \
                    + '_singleGate_' + str(self._single_gate)
        if self._single_gate is False:
            cell_str += '_gate_activation_' + self._gate_activation.__name__
        else:
            cell_str += '_single_gate_avg_' + str(self._single_gate_avg)
        return cell_str

    @property
    def state_size(self):
        return URNNStateTuple(self._num_units, self._num_units)

    @property
    def output_size(self):
        if self._num_proj is None:
            return self._num_units
        else:
            if self._complex_inout:
                return self._num_proj
            else:
                return self._num_proj

    def zero_state(self, batch_size, dtype=tf.float32):
        out = tf.zeros([batch_size, self.output_size], dtype=tf.float32)
        first_state = tf.zeros([batch_size, self._num_units])
        return URNNStateTuple(out, first_state)

    def single_memory_gate(self, h, x, scope, bias_init=0.0,
                           unitary=False, orthogonal=False):
        """
        New unified gate, idea use real and imaginary outputs as gating scalars.
        """
        with tf.variable_scope(scope, self._reuse):
            gh = complex_matmul(h, int(self._num_units/2.0), scope='gh', reuse=self._reuse,
                                unitary=unitary, orthogonal=orthogonal)
            gx = complex_matmul(x, int(self._num_units/2.0), scope='gx', reuse=self._reuse,
                                bias=True, bias_init_r=bias_init,
                                bias_init_c=bias_init)
            g = gh + gx
            if self._single_gate_avg:
                r = mod_sigmoid_beta(g, scope='r')
                z = mod_sigmoid_beta(g, scope='z')
                return r, z
            else:
                r = tf.nn.sigmoid(tf.real(g))
                z = tf.nn.sigmoid(tf.imag(g))
                return (tf.complex(r, tf.zeros_like(r), name='r'),
                        tf.complex(z, tf.zeros_like(z), name='z'))

    def double_memory_gate(self, h, x, scope, bias_init=4.0):
        """
        Complex GRU gates, the idea is that gates should make use of phase information.
        """
        with tf.variable_scope(scope, self._reuse):
            ghr = complex_matmul(h, int(self._num_units/2.0), scope='ghr', reuse=self._reuse)
            gxr = complex_matmul(x, int(self._num_units/2.0), scope='gxr', reuse=self._reuse,
                                 bias=True, bias_init_c=bias_init, bias_init_r=bias_init)
            gr = ghr + gxr
            r = self._gate_activation(gr, 'r', self._reuse)
            ghz = complex_matmul(h, int(self._num_units/2.0), scope='ghz', reuse=self._reuse)
            gxz = complex_matmul(x, int(self._num_units/2.0), scope='gxz', reuse=self._reuse,
                                 bias=True, bias_init_c=bias_init, bias_init_r=bias_init)
            gz = ghz + gxz
            z = self._gate_activation(gz, 'z', self._reuse)
            return r, z

    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope("ComplexGatedRecurrentUnit", reuse=self._reuse):
            _, last_h_real = state

            #assemble complex state
            last_h = tf.complex(last_h_real[:, :int(self._num_units/2)], 
                                last_h_real[:, int(self._num_units/2):] )

            if self._input_fourier:
                cinputs = tf.complex(inputs, tf.zeros_like(inputs))
                cin = tf.fft(cinputs)
            elif self._input_hilbert:
                cinputs = tf.complex(inputs, tf.zeros_like(inputs))
                cin = hilbert(cinputs)
            elif self._input_split_matmul:
                # Map the inputs from R to C.
                cinr = matmul_plus_bias(inputs, int(self._num_units/2.0), 'real', self._reuse)
                cini = matmul_plus_bias(inputs, int(self._num_units/2.0), 'imag', self._reuse)
                cin = tf.complex(cinr, cini)
            elif self._complex_inout:
                cin = inputs
            else:
                cin = tf.complex(inputs, tf.zeros_like(inputs))

            if self._single_gate:
                r, z = self.single_memory_gate(last_h, cin, 'memory_gate', bias_init=4.0,
                                               orthogonal=self._gateO)
            else:
                r, z = self.double_memory_gate(last_h, cin, 'double_memory_gate',
                                               bias_init=4.0)

            with tf.variable_scope("canditate_h"):
                in_shape = tf.Tensor.get_shape(cin).as_list()[-1]
                var_Wx = tf.get_variable("Wx", [in_shape, int(self._num_units/2.0), 2],
                                         dtype=tf.float32,
                                         initializer=tf.glorot_uniform_initializer())
                if self._stateU:
                    with tf.variable_scope("unitary_stiefel", reuse=self._reuse):
                        varU = tf.get_variable("recurrent_U",
                                               shape=[int(self._num_units/2.0),
                                                      int(self._num_units/2.0), 2],
                                               dtype=tf.float32,
                                               initializer=arjovski_init)
                        U = tf.complex(varU[:, :, 0], varU[:, :, 1])
                else:
                    varU = tf.get_variable("recurrent_U",
                                           shape=[int(self._num_units/2.0), 
                                                  int(self._num_units/2.0), 2],
                                           dtype=tf.float32,
                                           initializer=arjovski_init)
                    U = tf.complex(varU[:, :, 0], varU[:, :, 1])

                var_bias = tf.get_variable("b", [int(self._num_units/2.0), 2], dtype=tf.float32,
                                           initializer=tf.zeros_initializer())
                Wx = tf.complex(var_Wx[:, :, 0], var_Wx[:, :, 1])
                bias = tf.complex(var_bias[:, 0], var_bias[:, 1])
                tmp = tf.matmul(cin, Wx) + tf.matmul(tf.multiply(r, last_h), U) + bias
                h_bar = self._activation(tmp)
            new_h = (1 - z)*last_h + z*h_bar
            new_h_real = tf.concat([tf.real(new_h), tf.imag(new_h)], -1)

            if self._num_proj is None:
                output = new_h_real
            else:
                if self._complex_inout:
                    output = complex_matmul(new_h, self._num_proj, scope='C_to_C_out',
                                            reuse=self._reuse)
                    #disassemble complex state.
                    # output = tf.concat([tf.real(output), tf.imag(output)], -1)
                else:
                    output = C_to_R(new_h, self._num_proj, reuse=self._reuse)

            
            newstate = URNNStateTuple(output, new_h_real)
            return output, newstate
