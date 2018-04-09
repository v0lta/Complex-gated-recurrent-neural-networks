#!/usr/bin/env ipython
#
# Utility functions.
#
# ------------------------------------------
# author:       Stephanie Hyland (@corcra)
# date:         20/3/16
# ------------------------------------------

import tensorflow as tf
import numpy as np
import pdb

# tf 0.9
#from tf.nn.rnn_cell import RNNCell

# tf 0.7
#from tensorflow.models.rnn import rnn
#from tensorflow.models.rnn.rnn_cell import RNNCell
from tensorflow.python.ops import variable_scope as vs

from unitary import unitary

# === functions to help with implementing the theano version === #
# from http://arxiv.org/abs/1511.06464

def times_diag(arg, state_size, scope=None, real=False, split=False):
    """
    Multiplication with a diagonal matrix of the form exp(i theta_j)
    """
#    batch_size = arg.get_shape().as_list()[0]
    with vs.variable_scope(scope or "Times_Diag"):
        if not real:
            thetas = vs.get_variable("Thetas", 
                                     initializer=tf.constant(np.random.uniform(low=-np.pi, 
                                                                               high=np.pi, 
                                                                               size=state_size), 
                                                             dtype=tf.float32),
                                     dtype=tf.float32)
            # e(i theta)  = cos(theta) + i sin(theta)
            diagonal = tf.diag(tf.complex(tf.cos(thetas), tf.sin(thetas)))
            result = tf.matmul(arg, diagonal)
        else:
            # state is [state_re, state_im], remember
            hidden_size = state_size/2
            thetas = vs.get_variable("Thetas", 
                                     initializer=tf.constant(np.random.uniform(low=-np.pi, 
                                                                               high=np.pi, 
                                                                               size=hidden_size), 
                                                             dtype=tf.float32),
                                     dtype=tf.float32)
            diag_re = tf.diag(tf.cos(thetas))
            diag_im = tf.diag(tf.sin(thetas))
           
            if type(arg) == tuple:
                state_re = arg[0]
                state_im = arg[1]
            else:
                # cut it up
                state_re = tf.slice(arg, [0, 0], [-1, hidden_size])
                state_im = tf.slice(arg, [0, hidden_size], [-1, hidden_size])

            intermediate_re = tf.matmul(state_re, diag_re) - tf.matmul(state_im, diag_im)
            intermediate_im = tf.matmul(state_im, diag_re) + tf.matmul(state_re, diag_im)

            if split:
                result = intermediate_re, intermediate_im
            else:
                result = tf.concat(1, [intermediate_re, intermediate_im])
    return result

def reflection(state, state_size, scope=None, theano_reflection=True, real=False, split=False):
    """
    I do not entirely trust or believe the reflection operator in the theano version. :/
    TODO: indeed, it is wrong, wrongish.
    TODO: do the 'right' version.
    """
    # the reflections are initialised in a weird and tricky way: using initialize_matrix,
    # as if they are columns from a (2, state_size) matrix, so the range of random initialisation
    # is informed by both... but then my fixed_initializer function would return an incorrectly-sized
    # reflection, so I'm just going to do it manually.
    scale = np.sqrt(6.0/ (2 + state_size*2))
    
    with vs.variable_scope(scope or "Reflection"):
        reflect_re = vs.get_variable("Reflection/Real", dtype=tf.float32,
                                     initializer=tf.constant(np.random.uniform(low=-scale, high=scale, size=(state_size)),
                                                             dtype=tf.float32,
                                                             shape=[state_size, 1]))
        reflect_im = vs.get_variable("Reflection/Imaginary", dtype=tf.float32,
                                     initializer=tf.constant(np.random.uniform(low=-scale, high=scale, size=(state_size)),
                                                             dtype=tf.float32,
                                                             shape=[state_size, 1]))

        if real:
            # operation before this is fft, so...
            # ... not actually expecting a real state, isn't that fun
            state_re = tf.real(state)
            state_im = tf.imag(state)
            
            vstarv = tf.reduce_sum(reflect_re**2 + reflect_im**2)

            state_re_reflect_re = tf.matmul(state_re, reflect_re)
            state_re_reflect_im = tf.matmul(state_re, reflect_im)
            state_im_reflect_re = tf.matmul(state_im, reflect_re)
            state_im_reflect_im = tf.matmul(state_im, reflect_im)

            # tf.matmul with transposition is the same as T.outer
            # we need something of the shape [batch_size, state_size] in the end
            if theano_reflection:
                a = tf.matmul(state_re_reflect_re - state_im_reflect_im, reflect_re, transpose_b=True)
                b = tf.matmul(state_re_reflect_im + state_im_reflect_re, reflect_im, transpose_b=True)
                c = tf.matmul(state_re_reflect_re - state_im_reflect_im, reflect_im, transpose_b=True)
                d = tf.matmul(state_re_reflect_im + state_im_reflect_re, reflect_re, transpose_b=True)
                new_state_re = state_re - (2.0 / vstarv) * (a + b)
                new_state_im = state_im - (2.0 / vstarv) * (d - c)
            else:
                # TODO double-triple-check the maths here, blegh
                a = tf.matmul(state_re_reflect_re + state_im_reflect_im, reflect_re, transpose_b=True)
                b = tf.matmul(state_im_reflect_re - state_re_reflect_im, reflect_im, transpose_b=True)
                c = tf.matmul(state_im_reflect_re - state_re_reflect_im, reflect_re, transpose_b=True)
                d = tf.matmul(state_re_reflect_re + state_im_reflect_im, reflect_im, transpose_b=True)
                new_state_re = state_re - (2.0 / vstarv) * (a - b)
                new_state_im = state_im - (2.0 / vstarv) * (c + d)
            if split:
                return new_state_re, new_state_im
            else:
                return tf.concat(1, [new_state_re, new_state_im])

        elif theano_reflection:
            raise NotImplementedError
            # NOTE: I am *directly copying* what they do in the theano code (inasmuch as one can in TF),
            #       (s/input/state/g)
            # even though I think the maths might be incorrect, see this issue: https://github.com/amarshah/complex_RNN/issues/2
            # the function is times_reflection in models.py (not this file, hah hah hah!)

            state_re = tf.real(state)
            state_im = tf.imag(state)
            
            vstarv = tf.reduce_sum(reflect_re**2 + reflect_im**2)

            state_re_reflect_re = tf.matmul(state_re, reflect_re)
            state_re_reflect_im = tf.matmul(state_re, reflect_im)
            state_im_reflect_re = tf.matmul(state_im, reflect_re)
            state_im_reflect_im = tf.matmul(state_im, reflect_im)

            # tf.matmul with transposition is the same as T.outer
            # we need something of the shape [batch_size, state_size] in the end
            a = tf.matmul(state_re_reflect_re - state_im_reflect_im, reflect_re, transpose_b=True)
            b = tf.matmul(state_re_reflect_im + state_im_reflect_re, reflect_im, transpose_b=True)
            c = tf.matmul(state_re_reflect_re - state_im_reflect_im, reflect_im, transpose_b=True)
            d = tf.matmul(state_re_reflect_im + state_im_reflect_re, reflect_re, transpose_b=True)

            # the thing we return is:
            # return_re = state_re - (2/vstarv)(d - c)
            # return_im = state_im - (2/vstarv)(a + b)

            new_state_re = state_re - (2.0 / vstarv) * (a + b)
            new_state_im = state_im - (2.0 / vstarv) * (d - c)
            new_state = tf.complex(new_state_re, new_state_im)
            return new_state
        else:
            #TODO optimise I guess (SO MANY TRANSPOSE)
            vstarv = tf.reduce_sum(reflect_re**2 + reflect_im**2)
            prefactor = tf.complex(2.0/vstarv, 0.0)

            v = tf.complex(reflect_re, reflect_im)
            v_star = tf.conj(v)

            vx = tf.matmul(state, tf.reshape(v_star, [-1, 1]))
            new_state = state - prefactor *  tf.transpose(tf.matmul(v, tf.transpose(vx)))
            return new_state

def tanh_mod(x_vals, y_vals, scope=None, name=None):
    """
    tanh for complex-valued state
    (just applies it to the modulus of the state, leaves phase intact)
    ...
    assumes input is [batch size, 2*d]
    the second half of the columns are the imaginary parts
    """
    batch_size = x_vals.get_shape()[0]
    state_size = x_vals.get_shape()[1]
    hidden_size = state_size/2
    with vs.variable_scope(scope or "tanh_mod"):
        r = tf.sqrt(x_vals**2 + y_vals**2)
        r_scaled = tf.nn.tanh(r)
        # use half angle formula to get... angles
        # if y = 0...
        y_zeros = tf.equal(y_vals, 0)
        # if x > 0...
        x_g0 = tf.greater(x_vals, 0)
        x_l0 = tf.less(x_vals, 0)
        set_to_zero = tf.logical_and(y_zeros, x_g0)
        zero_matrix = tf.zeros_like(x_vals) + 1e-6
        set_to_pi = tf.logical_and(y_zeros, x_l0)
        pi_matrix = tf.mul(np.pi, tf.ones_like(x_vals))
        # get the values
        atan_arg = tf.div(r - x_vals, y_vals)
        pre_angle = 2*tf.atan(atan_arg)
        angle_filtered_zero = tf.select(set_to_zero, zero_matrix, pre_angle)
        angle = tf.select(set_to_pi, pi_matrix, angle_filtered_zero)
        # now recalculate the xes and ys
        x_scaled = tf.mul(r_scaled, tf.cos(angle))
        y_scaled = tf.mul(r_scaled, tf.sin(angle))
        output = tf.concat(1, [x_scaled, y_scaled], name=name)
    return output

def relu_mod(state, state_size, scope=None, real=False, name=None):
    """
    Rectified linear unit for complex-valued state.
    (Equation 8 in http://arxiv.org/abs/1511.06464)
    """
    batch_size = state.get_shape()[0]
    with vs.variable_scope(scope or "ReLU_mod"):
        if not real:
            # WARNING: complex_abs has no gradient registered in the docker version for some reason
            # [[ LookupError: No gradient defined for operation 'RNN/complex_RNN_99/ReLU_mod/ComplexAbs' (op type: ComplexAbs) ]]
            #modulus = tf.complex_abs(state)
            modulus = tf.sqrt(tf.real(state)**2 + tf.imag(state)**2)
            bias_term = vs.get_variable("Bias", dtype=tf.float32, 
                                        initializer=tf.constant(np.random.uniform(low=-0.01, high=0.01, size=(state_size)), 
                                                                dtype=tf.float32, 
                                                                shape=[state_size]))
                                        #        bias_tiled = tf.tile(bias_term, [1, batch_size])

            rescale = tf.complex(tf.maximum(modulus + bias_term, 0) / ( modulus + 1e-5*tf.ones_like(modulus)), tf.zeros_like(modulus))
                                        #rescale = tf.complex(tf.maximum(modulus + bias_term, 0) / ( modulus + 1e-5), 0.0)
        else:
            # state is [state_re, state_im]
            hidden_size = state_size/2
            state_re = tf.slice(state, [0, 0], [-1, hidden_size])
            state_im = tf.slice(state, [0, hidden_size], [-1, hidden_size])
            modulus = tf.sqrt(state_re**2 + state_im**2)
            # this is [batch_size, hidden_size] in shape, now...
            bias_re = vs.get_variable("Bias", dtype=tf.float32, 
                                      initializer=tf.constant(np.random.uniform(low=-0.01, high=0.01, size=(hidden_size)), 
                                                              dtype=tf.float32, 
                                                              shape=[hidden_size]))
            rescale = tf.maximum(modulus + bias_re, 0) / (modulus + 1e-5 * tf.ones_like(modulus))
#            bias_term = tf.concat(0, [bias_re, tf.zeros_like(bias_re)])
#            rescale = tf.maximum(modulus + bias_term, 0) / ( modulus + 1e-5*tf.ones_like(modulus) )
        output = tf.mul(state, tf.tile(rescale, [1, 2]), name=name)
    return output

def fixed_initializer(n_in_list, n_out, identity=-1, dtype=tf.float32):
    """
    This is a bit of a contrived initialiser to be consistent with the
    'initialize_matrix' function in models.py from the complex_RNN repo

    Basically, n_in is a list of input dimensions, because our linear map is
    folding together a bunch of linear maps, like:
        h = Ax + By + Cz + ...
    where x, y, z etc. might be different sizes
    so n_in is a list of [A.shape[0], B.shape[0], ...] in this example.
    and A.shape[1] == B.shape[1] == ...

    The resulting linear operator will have shape:
        ( sum(n_in), n_out )
    (because then one applies it to [x y z] etc.)

    The trick comes into it because we need A, B, C etc. to have initialisations
    which depend on their specific dimensions... their entries are sampled uniformly from
        sqrt(6/(in + out))
    
    So we have to initialise our special linear operator to have samples from different
    uniform distributions. Sort of gross, right? But it'll be fine.

    Finally: identity: what does it do?
    Well, it is possibly useful to initialise the weights associated with the internal state
    to be the identity. (Specifically, this is done in the IRNN case.)
    So if identity is >0, then it specifies which part of n_in_list (corresponding to a segment
    of the resulting matrix) should be initialised to identity, and not uniformly randomly as the rest.
    """
    matrix = np.empty(shape=(sum(n_in_list), n_out))
    row_marker = 0
    for (i, n_in) in enumerate(n_in_list):
        if i == identity:
            values = np.identity(n_in)
        else:
            scale = np.sqrt(6.0/ (n_in + n_out))
            values = np.asarray(np.random.uniform(low=-scale, high=scale,
                                                  size=(n_in, n_out)))
        # NOTE: HARDCODED DTYPE
        matrix[row_marker:(row_marker + n_in), :] = values
        row_marker += n_in
    return tf.constant(matrix, dtype=dtype)

def IRNN_initializer(n_in_list, n_out, identity=-1, dtype=tf.float32):
    """
    sort of like fixed_initialiser, but random gaussian for non-identity part
    (closer to description in Le et al, 2015)
    """
    matrix = np.empty(shape=(sum(n_in_list), n_out))
    row_marker = 0
    for (i, n_in) in enumerate(n_in_list):
        if i == identity:
            values = np.identity(n_in)
        else:
            values = np.asarray(np.random.normal(loc=0.0, scale=0.001,
                                                 size=(n_in, n_out)))
        # NOTE: HARDCODED DTYPE
        matrix[row_marker:(row_marker + n_in), :] = values
        row_marker += n_in
    return tf.constant(matrix, dtype=dtype)

# === more generic functions === #
def linear(args, output_size, bias, bias_start=0.0, 
           scope=None, identity=-1, dtype=tf.float32, init_val=None):
    """
    variant of linear from tensorflow/python/ops/rnn_cell
    ... variant so I can specify the initialiser!

    Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.
    Args:
        args:           a 2D Tensor or a list of 2D, batch x n, Tensors.
        output_size:    int, second dimension of W[i].
        bias:           boolean, whether to add a bias term or not.
        bias_start:     starting value to initialize the bias; 0 by default.
        scope:          VariableScope for the created subgraph; defaults to "Linear".
        identity:       which matrix corresponding to inputs should be initialised to identity?
        dtype:          data type of linear operators
        init_val:       optional matrix to use as fixed initialiser
    Returns:
        A 2D Tensor with shape [batch x output_size] equal to
        sum_i(args[i] * W[i]), where W[i]s are newly created matrices.
    Raises:
        ValueError: if some of the arguments has unspecified or wrong shape.
  """
    assert args is not None
    if not isinstance(args, (list, tuple)):
        args = [args]

    # Calculate the total size of arguments on dimension 1.
    total_arg_size = 0
    shapes = [a.get_shape().as_list() for a in args]
    n_in_list = []
    for shape in shapes:
        if len(shape) != 2:
            raise ValueError("Linear is expecting 2D arguments: %s" % str(shapes))
        if not shape[1]:
            raise ValueError("Linear expects shape[1] of arguments: %s" % str(shapes))
        else:
            total_arg_size += shape[1]
            n_in_list.append(shape[1])

    # Now the computation.
    with vs.variable_scope(scope or "Linear"):
        if init_val is None:
            if identity > 0:
                # IRNN setting
                matrix = vs.get_variable("Matrix", dtype=dtype, initializer=IRNN_initializer(n_in_list, output_size, identity, dtype))
            else:
                # not IRNN
                matrix = vs.get_variable("Matrix", dtype=dtype, initializer=fixed_initializer(n_in_list, output_size, identity, dtype))
        else:
            matrix = vs.get_variable("Matrix", dtype=dtype, initializer=tf.constant(init_val, dtype=dtype))
        if len(args) == 1:
            res = tf.matmul(args[0], matrix)
        else:
            res = tf.matmul(tf.concat(1, args), matrix)
        if not bias:
            return res
        bias_term = vs.get_variable("Bias", dtype=dtype, initializer=tf.constant(bias_start, dtype=dtype, shape=[output_size]))
    return res + bias_term

def linear_complex(arg_re, arg_im, output_size, bias, bias_start=0.0, 
           scope=None, identity=-1, dtype=tf.float32,
           init_val_re=None, init_val_im=None):
    """
    NOTE: arg is a single arg, because that's how it is

    Linear map:
                arg * W where W is complex, but really:
            W_re * arg_re - W_im * arg_im + i (W_re * arg_im + W_im * arg_re)
                where W_re, W_im are Variables
    Args:
        arg:            a 2D Tensor (batch x n))))
        output_size:    int, second dimension of W[i].
        bias:           boolean, whether to add a bias term or not.
        bias_start:     starting value to initialize the bias; 0 by default.
        scope:          VariableScope for the created subgraph; defaults to "LinearComplex".
        identity:       which matrix corresponding to inputs should be initialised to identity?
        dtype:          data type of linear operators

    Returns:
        2D tensor with shape [batch x output_size] equal to arg * W
    Raises:
        ValueError: if some of the arguments has unspecified or wrong shape.
    """
    shape = arg_re.get_shape().as_list()
    if len(shape) != 2:
        raise ValueError("LinearComplex is expecting a 2D argument")
    n = shape[1]

    # Now the computation.
    with vs.variable_scope(scope or "LinearComplex"):
        if init_val_re is None:
            matrix_re = vs.get_variable("Matrix/Real", dtype=dtype, initializer=fixed_initializer([n], output_size, identity, dtype))
        else:
            matrix_re = vs.get_variable("Matrix/Real", dtype=dtype, initializer=tf.constant(init_val_re, dtype=dtype))
        if init_val_im is None:
            matrix_im = vs.get_variable("Matrix/Imaginary", dtype=dtype, initializer=fixed_initializer([n], output_size, identity, dtype))
        else:
            matrix_im = vs.get_variable("Matrix/Imaginary", dtype=dtype, initializer=tf.constant(init_val_im, dtype=dtype))

        # HERE IT IS!
        res_re = tf.matmul(arg_re, matrix_re) - tf.matmul(arg_im, matrix_im)
        res_im = tf.matmul(arg_im, matrix_re) + tf.matmul(arg_re, matrix_im)
        # there's the end fo that bit

        if not bias:
            return res_re, res_im
        else:
            bias_re = vs.get_variable("Bias/Real", dtype=dtype, initializer=tf.constant(bias_start, dtype=dtype, shape=[output_size]))
            bias_im = vs.get_variable("Bias/Imaginary", dtype=dtype, initializer=tf.constant(bias_start, dtype=dtype, shape=[output_size]))
            return res_re + bias_re, res_im + bias_im

# === RNNs ! === #

def RNN(cell_type, x, input_size, state_size, output_size, sequence_length, init_re=None, init_im=None):
    batch_size = tf.shape(x)[0]
    if cell_type == 'tanhRNN':
        cell = tanhRNNCell(input_size=input_size, state_size=state_size, output_size=output_size, state_dtype=x.dtype)
        state_0 = cell.zero_state(batch_size)
    elif cell_type == 'IRNN':
        cell = IRNNCell(input_size=input_size, state_size=state_size, output_size=output_size, state_dtype=x.dtype)
        state_0 = cell.zero_state(batch_size)
    elif cell_type == 'LSTM':
        cell = LSTMCell(input_size=input_size, state_size=2*state_size, output_size=output_size, state_dtype=x.dtype)
        state_0 = cell.zero_state(batch_size)
    elif cell_type == 'complex_RNN':
        cell = complex_RNNCell(input_size=input_size, state_size=2*state_size, output_size=output_size, state_dtype=x.dtype)
        state_0 = cell.h0(batch_size)
    elif cell_type == 'uRNN':
        #cell = uRNNCell(input_size=input_size, state_size=state_size, output_size=output_size, state_dtype=tf.complex64, init_re=init_re, init_im=init_im)
        cell = uRNNCell(input_size=input_size, state_size=2*state_size, output_size=output_size, state_dtype=x.dtype, init_re=init_re, init_im=init_im)
        state_0 = cell.h0(batch_size)
    elif cell_type == 'ortho_tanhRNN':
        cell = tanhRNNCell(input_size=input_size, state_size=state_size, output_size=output_size, state_dtype=x.dtype)
    elif cell_type == 'LT-ORNN':
        cell = LTRNNCell(input_size=input_size, state_size=state_size, output_size=output_size, state_dtype=x.dtype, orthogonal=True)
        state_0 = cell.zero_state(batch_size)
    elif cell_type == 'LT-IRNN':
        cell = LTRNNCell(input_size=input_size, state_size=state_size, output_size=output_size, state_dtype=x.dtype, orthogonal=False)
        state_0 = cell.zero_state(batch_size)
    else: 
        raise NotImplementedError
    # split up the input so the RNN can accept it...
    # TODO DEBUG TESTING
    if input_size > 1:
        inputs = [tf.squeeze(input_, [1])
                for input_ in tf.split(1, sequence_length, x)]
    else:
        inputs = tf.split(1, sequence_length, x)
    # tf 0.9.0
    outputs, final_state = tf.nn.rnn(cell, inputs, initial_state=state_0)
    return outputs

# === cells ! === #
# TODO: better name for this abstract class
#class steph_RNNCell(RNNCell):      # tf 0.7.0
class steph_RNNCell(tf.nn.rnn_cell.RNNCell):            # tf 0.9.0
    def __init__(self, input_size, state_size, output_size, state_dtype, 
                 init_re=None, init_im=None, orthogonal=False):
        self._input_size = input_size
        self._state_size = state_size
        self._output_size = output_size
        self._state_dtype = state_dtype
        self._init_re = init_re
        self._init_im = init_im
        self._orthogonal = orthogonal

    @property
    def input_size(self):
        return self._input_size

    @property
    def output_size(self):
        return self._output_size

    @property
    def state_size(self):
        return self._state_size

    @property
    def state_dtype(self):
        return self._state_dtype

    def zero_state(self, batch_size, dtype=None):
        """
        Return state tensor (shape [batch_size x state_size]) filled with 0.

        Args:
            batch_size:     int, float, or unit Tensor representing the batch size.
            dtype:          the data type to use for the state
                            (optional, if None use self.state_dtype)
        Returns:
            A 2D Tensor of shape [batch_size x state_size] filled with zeros.
        """
        if dtype is None:
            dtype = self.state_dtype
        zeros = tf.zeros(tf.pack([batch_size, self._state_size]), dtype=dtype)
        zeros.set_shape([None, self._state_size])
        return zeros

    def h0(self, batch_size, dtype=None):
        """
        Return state tensor (shape [batch_size x state_size]) filled with rv ~ unif(-sqrt(3/state_size), sqrt(3/state_size)

        Args:
            batch_size:     int, float, or unit Tensor representing the batch size.
            state size:     int of the dimension of internal state (may be 2x hidden in complex case)
            dtype:          the data type to use for the state
                            (optional, if None use self.state_dtype)
        Returns:
            A 2D Tensor of shape [batch_size x state_size] filled with rv ~ unif(...)
        """
        if dtype is None:
            dtype = self.state_dtype
        bucket = np.sqrt(3.0/self._state_size)
        first_state = tf.random_uniform([batch_size, self._state_size], minval=-bucket, maxval=bucket, dtype=dtype)
        return first_state

    def __call__(self):
        """
        Run this RNN cell on inputs, starting from the given state.
        
        Args:
            inputs: 2D Tensor with shape [batch_size x self.input_size].
            state: 2D Tensor with shape [batch_size x self.state_size].
            scope: VariableScope for the created subgraph; defaults to class name.
       
       Returns:
            A pair containing:
            - Output: A 2D Tensor with shape [batch_size x self.output_size]
            - New state: A 2D Tensor with shape [batch_size x self.state_size].
        """
        raise NotImplementedError("Abstract method")

class tanhRNNCell(steph_RNNCell):
    def __call__(self, inputs, state, scope='tanhRNN'):
        """ 
        Slightly-less-basic RNN: 
            state = tanh(linear(previous_state, input))
            output = linear(state)
        """
        with vs.variable_scope(scope):
            new_state = tf.tanh(linear([inputs], self._state_size, bias=True, scope='Linear/FoldIn') + linear([state], self._state_size, bias=False, scope='Linear/Transition'), name='new_state')
            output = linear(new_state, self._output_size, bias=True, scope='Linear/Output')
        return output, new_state

class IRNNCell(steph_RNNCell):
    def __call__(self, inputs, state, scope='IRNN'):
        """ 
        Slightly-less-basic RNN: 
            state = relu(linear(previous_state, input))
            output = linear(state)
        ... but the state linear is initialised in a special way!
        """
        with vs.variable_scope(scope):
            # the identity flag says we initialize the part of the transition matrix corresponding to the 1th element of
            # the first input to linear (a.g. [inputs, state], aka 'state') to the identity
            new_state = tf.nn.relu(linear([inputs, state], self._state_size, bias=True, scope='Linear/Transition', identity=1), name='new_state')
            output = linear(new_state, self._output_size, bias=True, scope='Linear/Output')
        return output, new_state

class LTRNNCell(steph_RNNCell):
    def __call__(self, inputs, state, scope='LTRNN'):
        """ 
        Linear transition RNN as in
            https://arxiv.org/abs/1602.06662
            "Orthogonal RNNs and Long-Memory Tasks", Henaff, szlam, LeCun

            state = sigmoid(linear(input) + bias) + linear(state)
            output = linear(state)
        """
        with vs.variable_scope(scope):
            if self._orthogonal:
                # generate an orthogonal matrix to initialise RNN with
                matrix = np.random.normal(size=(self._state_size, self._state_size))
                init_V, _ = np.linalg.qr(matrix, mode='complete')
                assert np.allclose(np.dot(init_V.T, init_V), np.identity(self._state_size))
            else:
                # initialise with identity
                init_V = np.identity(self._state_size)
            new_state = tf.nn.sigmoid(linear(inputs, self._state_size, bias=True, scope='Linear/FoldIn')) + linear(state, self._state_size, bias=False, scope='Linear/Transition', init_val=init_V)
            output = linear(new_state, self._output_size, bias=False, scope='Linear/Output')
        return output, new_state

class LSTMCell(steph_RNNCell):
    def __call__(self, inputs, state, scope='LSTM'):
        """
        Inspired by LSTMCell in tensorflow (python/ops/rnn_cell), but modified to
        be consistent with the version in the Theano implementation. (There are small
        differences...)
        """
        # the state is actually composed of both hidden and state parts:
        # (so they're each half the size of the state, which will be doubled elsewhere)
        # that is confusing nomenclature, I realise
        hidden_size = self._state_size/2
        state_prev = tf.slice(state, [0, 0], [-1, hidden_size])
        hidden_prev = tf.slice(state, [0, hidden_size], [-1, hidden_size])

        with vs.variable_scope(scope):
            i = tf.sigmoid(linear([inputs, hidden_prev], hidden_size, bias=True, scope='Linear/Input'))
            candidate = tf.tanh(linear([inputs, hidden_prev], hidden_size, bias=True, scope='Linear/Candidate'))
            forget = tf.sigmoid(linear([inputs, hidden_prev], hidden_size, bias=True, scope='Linear/Forget'))
            
            intermediate_state = tf.add(tf.mul(i, candidate), tf.mul(forget, state_prev), name='new_state')
            
            # out (not the real output, confusingly)
            # NOTE: this differs from the LSTM implementation in TensorFlow
            # in tf, the intermediate_state doesn't contribute
            out = tf.sigmoid(linear([inputs, hidden_prev, intermediate_state], hidden_size, bias=True, scope='Linear/Out'))
       
            intermediate_hidden = out * tf.tanh(intermediate_state)
            
            # now for the 'actual' output
            output = linear([intermediate_hidden], self._output_size, bias=True, scope='Linear/Output')
            
            # the 'state' to be fed back in (to be split up, again!)
            new_state = tf.concat(1, [intermediate_state, intermediate_hidden])
        return output, new_state

class complex_RNNCell(steph_RNNCell):
    def __call__(self, inputs, state, scope='complex_RNN', real=True):
        """
        (copying their naming conventions, mkay)
        """
        # TODO: set up data types at time of model selection
        # (for now:) cast inputs to complex
        with vs.variable_scope(scope):
            if not real:
                step1 = times_diag(state, self._state_size, scope='Diag/First')
                step2 = tf.batch_fft(step1, name='FFT')
#                step3 = reflection(step2, self._state_size, scope='Reflection/First')
                step3 = step2
                permutation = vs.get_variable("Permutation", dtype=tf.complex64, 
                                              initializer=tf.complex(np.random.permutation(np.eye(self._state_size, dtype=np.float32)), tf.constant(0.0, dtype=tf.float32)),
                                              trainable=False)
                step4 = tf.matmul(step3, permutation)
                step5 = times_diag(step4, self._state_size, scope='Diag/Second')
                step6 = tf.batch_ifft(step5, name='InverseFFT')
                step7 = reflection(step6, self._state_size, scope='Reflection/Second')
                step8 = times_diag(step7, self._state_size, scope='Diag/Third')
                step1 = state

                # (folding in the input data) 
                foldin_re = linear(inputs, self._state_size, bias=False, scope='Linear/FoldIn/Real')
                foldin_im = linear(inputs, self._state_size, bias=False, scope='Linear/FoldIn/Imaginary')
                intermediate_state = tf.complex(foldin_re, foldin_im, name='Linear/Intermediate/Complex') + step8
#                intermediate_re = foldin_re + tf.real(step8)
#                intermediate_im = foldin_im + tf.imag(step8)
#                intermediate_state = tf.concat(1, [intermediate_re, intermediate_im])
#                new_state_real = relu_mod(intermediate_state, self._state_size, scope='ReLU_mod', real=True)
                
                new_state = relu_mod(intermediate_state, self._state_size, scope='ReLU_mod', name='new_state')     # DEFAULT
                #new_state = tf.nn.relu(intermediate_state, name='new_state')
                #new_state = tf.nn.tanh(intermediate_state, name='new_state')
                #new_state = tf.identity(intermediate_state, name='new_state')
                #new_state = tf.maximum(0.1*intermediate_state, intermediate_state, name='new_state')       # ... leaky relu
#                new_state = intermediate_state
           
            # taken from uRNN
            #new_state = tanh_mod(intermediate_re, intermediate_im, scope='tanh_mod', name='new_state')
            #new_state = tf.nn.sigmoid(intermediate_state, name='new_state')
            #new_state = relu_mod(intermediate_state, self._state_size, scope='ReLU_mod', real=True, name='new_state')

                real_state = tf.concat(1, [tf.real(new_state), tf.imag(new_state)])
                output = linear(real_state, self._output_size, bias=True, scope='Linear/Output')
#                output = linear(new_state_real, self._output_size, bias=True, scope='Linear/Output')

#                new_state_re = tf.slice(new_state_real, [0, 0], [-1, self._state_size])
#                new_state_im = tf.slice(new_state_real, [0, self._state_size], [-1, self._state_size])
#                new_state = tf.complex(new_state_re, new_state_im)
            else:
                # state is [state_re, state_im]
                step1_re, step1_im = times_diag(state, self._state_size, scope='Diag/First', real=True, split=True)             # gradient is fine
                step2 = tf.div(tf.batch_fft(tf.complex(step1_re, step1_im)), np.sqrt(self._state_size/2), name='FFT')            # gradient looks ok
                step3_re, step3_im = reflection(step2, self._state_size/2, scope='Reflection/First', real=True, split=True)      # gradient looks ok, but is it giving the right reflection formula?
                permutation = vs.get_variable("Permutation", dtype=tf.float32, 
                                              initializer=tf.constant(np.random.permutation(np.eye(self._state_size/2, dtype=np.float32))),
                                              trainable=False)
                step4_re = tf.matmul(step3_re, permutation)
                step4_im = tf.matmul(step3_im, permutation)
#                step4 = tf.concat(1, [step4_re, step4_im])
                step5_re, step5_im = times_diag((step4_re, step4_im), self._state_size, scope='Diag/Second', real=True, split=True)
                step6 = tf.mul(tf.batch_ifft(tf.complex(step5_re, step5_im)), np.sqrt(self._state_size/2), name='IFFT')
                step7 = reflection(step6, self._state_size/2, scope='Reflection/Second', real=True, split=False)
                step8 = times_diag(step7, self._state_size, scope='Diag/Third', real=True, split=False)
                
                foldin = linear(inputs, self._state_size, bias=False, scope='Linear/FoldIn/Real')
                
                intermediate_state = step8 + foldin

                new_state = relu_mod(intermediate_state, self._state_size,scope='ReLU_mod', real=True, name='new_state')
                
                output = linear(new_state, self._output_size, bias=True, scope='Linear/Output')

        return output, new_state

class uRNNCell(steph_RNNCell):
    def __call__(self, inputs, state, scope='uRNN', split=True):
        """
        this unitary RNN shall be my one, once I figure it out I guess
        ... fun times ahead
        """
        with vs.variable_scope(scope):
            # transform the hidden state
            hidden_size = self._state_size/2
            state_re = tf.slice(state, [0, 0], [-1, hidden_size])
            state_im = tf.slice(state, [0, hidden_size], [-1, hidden_size])

#            Ustate_re = linear(state_re, hidden_size, bias=True, scope='Unitary/Transition/Real', init_val=self._init_re)
#            Ustate_im = linear(state_im, hidden_size, bias=True, scope='Unitary/Transition/Imaginary', init_val=self._init_im)
    
            BETA = 1.05     # accounting for the effects of the nonlinearity (EXPERIMENTAL)
            Ustate_re, Ustate_im = linear_complex(state_re, state_im, hidden_size, bias=False, scope='Unitary/Transition', init_val_re=self._init_re, init_val_im=self._init_im)
#                    Ustate_im = linear(state_im, hidden_size, bias=True, scope='Unitary/Transition/Imaginary', init_val=self._init_im)
            foldin_re = linear(inputs, hidden_size, bias=True, scope='Linear/FoldIn/Real')
            foldin_im = linear(inputs, hidden_size, bias=True, scope='Linear/FoldIn/Imaginary')
            intermediate_re = foldin_re + BETA*Ustate_re
            intermediate_im = foldin_im + BETA*Ustate_im

            intermediate_state = tf.concat(1, [intermediate_re, intermediate_im])
          
            # identity
            #new_state = tf.identity(intermediate_state, name='new_state')
            #new_state = tanh_mod(intermediate_re, intermediate_im, scope='tanh_mod', name='new_state')
            new_state = tf.nn.tanh(intermediate_state, name='new_state')
            #new_state = tf.nn.relu(intermediate_state, name='new_state')
            #new_state = tf.nn.sigmoid(intermediate_state, name='new_state')
            #new_state = tf.maximum(0.1*intermediate_state, intermediate_state, name='new_state')       # ... leaky relu
            #new_state = relu_mod(intermediate_state, self._state_size, scope='ReLU_mod', real=True, name='new_state')

            output = linear(new_state, self._output_size, bias=True, scope='Linear/Output')
        return output, new_state
