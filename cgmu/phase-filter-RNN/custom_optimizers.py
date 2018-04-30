"""
    Implementation of unitary and orthogonality preserving
    tensorflow optimization ops.
"""
import tensorflow as tf
from IPython.core.debugger import Tracer
debug_here = Tracer()
from tensorflow.python.training import training_ops


class RMSpropNatGrad(tf.train.Optimizer):
    """ RMSProp optimizer with the capability to do natural gradient steps.
        Inspired by: https://github.com/stwisdom/urnn/blob/master/custom_optimizers.py
        See also:
        Full-Capacity Unitary Recurrent Neural Networks, Wisdom et al, at:
        https://arxiv.org/abs/1611.00035
    """

    def __init__(self, learning_rate, decay=0.9, momentum=0.0,
                 epsilon=1e-10, nat_grad_normalization=False,
                 name='RMSpropNatGrad'):
        """
            TODO: Do documentation.
            TODO: Implement unitary momentum.
        """
        use_locking = False
        super().__init__(use_locking, name)
        self._learning_rate = learning_rate
        self._decay = decay
        self._momentum = momentum
        self._epsilon = epsilon
        self._nat_grad_normalization = nat_grad_normalization
        self._debug = True

        # Tensors for learning rate and momentum.  Created in _prepare.
        self._learning_rate_tensor = None
        self._decay_tensor = None
        self._momentum_tensor = None
        self._epsilon_tensor = None

        print("training params:", self._learning_rate, self._decay, self._momentum)

    def _create_slots(self, var_list):
        """ Set up rmsprop slots for all variables."""
        for v in var_list:
            init_rms = tf.ones_initializer(dtype=v.dtype)
            self._get_or_make_slot_with_initializer(v, init_rms, v.get_shape(),
                                                    v.dtype, "rms", self._name)
            self._zeros_slot(v, "momentum", self._name)
            init_eps = tf.constant_initializer(self._epsilon)
            self._get_or_make_slot_with_initializer(v, init_eps, v.get_shape(),
                                                    v.dtype, "eps", self._name)

    def _prepare(self):
        """Convert algorthm parameters to tensors. """
        self._learning_rate_tensor = tf.convert_to_tensor(self._learning_rate,
                                                          name="learning_rate")
        self._decay_tensor = tf.convert_to_tensor(self._decay, name="decay")
        self._momentum_tensor = tf.convert_to_tensor(self._momentum,
                                                     name="momentum")
        self._epsilon_tensor = tf.convert_to_tensor(self._epsilon,
                                                    name="epsilon")

    def _summary_A(self, A):
        # test A's skew symmetrie:
        test_a = tf.transpose(tf.conj(A)) - (-A)
        test_a_norm = tf.real(tf.norm(test_a))
        tf.summary.scalar('A.H--A', test_a_norm)

    def _summary_C(self, C):
        # C must be unitary/orthogonal:
        eye = tf.eye(*tf.Tensor.get_shape(C).as_list(), dtype=C.dtype)
        test_c = eye - tf.matmul(tf.transpose(tf.conj(C)), C)
        test_c_norm = tf.real(tf.norm(test_c))
        tf.summary.scalar('I-C.HC', test_c_norm)

    def _summary_W(self, W):
        # W must also be unitary/orthogonal:
        eye = tf.eye(*tf.Tensor.get_shape(W).as_list(), dtype=W.dtype)
        test_w = eye - tf.matmul(tf.transpose(tf.conj(W)), W)
        test_w_norm = tf.real(tf.norm(test_w))
        tf.summary.scalar('I-W.HW', test_w_norm)

    def _apply_dense(self, grad, var):
        rms = self.get_slot(var, "rms")
        mom = self.get_slot(var, "momentum")
        eps = self.get_slot(var, 'eps')
        # debug_here()
        if 'orthogonal_stiefel' in var.name and 'bias' not in var.name:
            with tf.variable_scope("orthogonal_step"):
                print('Appling an orthogonality preserving step to', var.name)
                # apply the rms update rule.
                new_rms = self._decay_tensor * rms + (1. - self._decay_tensor) \
                    * tf.square(grad)
                rms_assign_op = tf.assign(rms, new_rms)
                # scale the gradient.
                if self._nat_grad_normalization:
                    grad = grad / (tf.sqrt(rms) + eps)
                # the update should preserve orthogonality.
                grad_shape = tf.Tensor.get_shape(grad).as_list()
                # W_new_lst = []
                eye = tf.eye(grad_shape[0], dtype=tf.float32)
                G = grad
                W = var
                A = tf.matmul(tf.transpose(G), W) - tf.matmul(tf.transpose(W), G)
                cayleyDenom = eye + (self._learning_rate_tensor/2.0 * A)
                cayleyNumer = eye - (self._learning_rate_tensor/2.0 * A)
                C = tf.matmul(tf.matrix_inverse(cayleyDenom), cayleyNumer)
                W_new = tf.matmul(C, W)
                if self._debug:
                    self._summary_A(A)
                    self._summary_C(C)
                    self._summary_W(W)
                var_update_op = tf.assign(var, W_new)
                return tf.group(*[var_update_op, rms_assign_op])
        elif 'unitary_stiefel' in var.name and 'bias' not in var.name:
            with tf.variable_scope("unitary_step"):
                print('Appling an unitarity preserving step to', var.name)
                # apply the rms update rule.
                new_rms = self._decay_tensor * rms + (1. - self._decay_tensor) \
                    * tf.square(grad)
                rms_assign_op = tf.assign(rms, new_rms)
                # scale the gradient.
                if self._nat_grad_normalization:
                    grad = grad / (tf.sqrt(new_rms) + eps)
                # do an update step, which preserves unitary structure.
                # checking shapes.
                grad_shape = tf.Tensor.get_shape(grad).as_list()
                assert grad_shape[0] == grad_shape[1]
                eye = tf.eye(grad_shape[0], dtype=tf.complex64)
                G = tf.complex(grad[:, :, 0], grad[:, :, 1])
                W = tf.complex(var[:, :, 0], var[:, :, 1])
                A = tf.matmul(tf.conj(tf.transpose(G)), W) \
                    - tf.matmul(tf.conj(tf.transpose(W)), G)
                # A must be skew symmetric.
                larning_rate_scale = tf.complex(self._learning_rate_tensor/2.0,
                                                tf.zeros_like(self._learning_rate_tensor))
                cayleyDenom = eye + larning_rate_scale * A
                cayleyNumer = eye - larning_rate_scale * A
                C = tf.matmul(tf.matrix_inverse(cayleyDenom), cayleyNumer)
                W_new = tf.matmul(C, W)
                if self._debug:
                    self._summary_A(A)
                    self._summary_C(C)
                    self._summary_W(W)
                W_new_re = tf.real(W_new)
                W_new_img = tf.imag(W_new)
                W_array = tf.stack([W_new_re, W_new_img], -1)
                var_update_op = tf.assign(var, W_array)
                return tf.group(*[var_update_op, rms_assign_op])
        else:
            # do the usual RMSprop update
            if 0:
                # tensorflow default.
                print('Appling standard rmsprop to', var.name)
                return training_ops.apply_rms_prop(
                    var, rms, mom,
                    tf.cast(self._learning_rate_tensor, var.dtype.base_dtype),
                    tf.cast(self._decay_tensor, var.dtype.base_dtype),
                    tf.cast(self._momentum_tensor, var.dtype.base_dtype),
                    tf.cast(self._epsilon_tensor, var.dtype.base_dtype),
                    grad, use_locking=False).op
            else:
                # My rmsprop implementation.
                new_rms = self._decay_tensor * rms \
                    + (1. - self._decay_tensor) * tf.square(grad)
                rms_assign_op = tf.assign(rms, new_rms)
                W_new = var - self._learning_rate_tensor * grad / (tf.sqrt(new_rms) + eps)
                var_update_op = tf.assign(var, W_new)
                return tf.group(*[var_update_op, rms_assign_op])

    def _apply_sparse(self, grad, var):
        raise NotImplementedError("Sparse gradient updates are not supported.")
