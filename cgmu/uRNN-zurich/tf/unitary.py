#!/usr/bin/env ipython
#
# Functions pertaining to the unitary group and its associated Lie algebra.
# (RETURN TF OBJECTS)
#
# ------------------------------------------
# author:       Stephanie Hyland (@corcra)
# date:         12/4/16
# ------------------------------------------

import tensorflow as tf
import numpy as np
import pdb

from tensorflow.python.ops import variable_scope as vs

from unitary_np import lie_algebra_basis_element

def lie_algebra_element(n, lambdas, sparse=False):
    """
    Explicitly construct an element of a Lie algebra, assuming 'default' basis,
    given a set of coefficients (lambdas).

    That is,
        L = sum lambda_i T^i
    where T^i are the basis elements and lambda_i are the coefficients

    The Lie algebra is u(n), associated to the Lie group U(n) of n x n unitary matrices.
    The Lie algebra u(n) is n x n skew-Hermitian matrices. 

    Args:
        n:              see above
        lambdas:        a Tensor of length n x n
    Returns:
        a 2D Tensor of shape [n, n], dtype tf.complex64, the element of the Lie algebra

    This incremental sparse idea is taken from 
        http://stackoverflow.com/questions/34685947/adjust-single-value-within-tensor-tensorflow

    POSSIBLE TODO: combine this with function to generate elements of the basis,
        or create a basis-element generator
    """
    lie_algebra_dim = n*n
    assert lambdas.get_shape().as_list() == [1, lie_algebra_dim]

    flat_lambdas = tf.squeeze(lambdas)
    L_shape = [n, n]

    if sparse:
        print 'ERROR: This function requires gradient for SparseToDense. Does not yet exist.'
        raise NotImplementedError

        # === useful things === #
        indices_diag = [[i, i] for i in xrange(n)]
        indices_upper = [[i, j] for i in xrange(n) for j in xrange(i + 1, n)]
        indices_lower = [[j, i] for i in xrange(n) for j in xrange(i + 1, n)]

        # === construct two sparse tensors! === #
        # == imaginary == #
        # all positive, so we don't need to change the values at all
        # values along diagonal
        values_im_diag = tf.slice(flat_lambdas, [0], [n])
        # off-diagonal values (twice)
        values_im_rest = tf.slice(flat_lambdas, [n], [(n * (n - 1) / 2)])
        values_im = tf.concat(0, [values_im_diag, values_im_rest, values_im_rest])
        # indices, now (this is not the most efficient or elegant but it mirrors basis-creation function)
        indices_im = indices_diag + indices_upper + indices_lower
        # create tensor
        L_im_sparse = tf.SparseTensor(indices_im, values_im, L_shape)
        L_im_reorder = tf.sparse_reorder(L_im_sparse)
        # NOTE: ERROR: No gradient defined for operation ... (op type: SparseToDense)
        L_im = tf.sparse_tensor_to_dense(L_im_reorder)

        # == real == #
        # need positive and negative versions of the values
        values_re_pos = tf.slice(flat_lambdas, [n + (n * (n - 1) / 2)], [n * (n - 1) / 2])
        values_re = tf.concat(0, [values_re_pos, tf.neg(values_re_pos)])
        # have to make sure the neg and pos values get the correct indices!
        indices_re = indices_upper + indices_lower
        # create tensor
        L_re_sparse = tf.SparseTensor(indices_re, values_re, L_shape)
        L_re_reorder = tf.sparse_reorder(L_re_sparse)
        # NOTE: ERROR: No gradient defined for operation ... (op type: SparseToDense)
        L_re = tf.sparse_tensor_to_dense(L_re_reorder)
    else:
        # *without* sparse tensors!
        # == init == #
        L_re = tf.zeros(shape=[n, n], dtype=tf.float32)
        L_im = tf.zeros(shape=[n, n], dtype=tf.float32)

        # == run through == #
        # (this must be so immensely inefficient ;__; )
        for e in xrange(0, lie_algebra_dim):
            T_re, T_im = lie_algebra_basis_element(n, e)
            temp_re = tf.mul(flat_lambdas[e], tf.constant(T_re))
            L_re += temp_re

            temp_im = tf.mul(flat_lambdas[e], tf.constant(T_im))
            L_im += temp_im

    # TODO: test/prove that these produce identical results

    # === combine! ==== #
    L = tf.complex(L_re, L_im)
    return L

def unitary(arg, state_size, scope=None):
    """
    A linear map:
        arg * U, where U = expm(sum_i lambda_i T_i)
    where lambda_i are variables and T_i are constant Tensors given by basis
    U is square, n x n where n == state_size

    Args:
        arg: a 2D Tensor, batch x n (NOTE: only allowing one, for now)
            note: n must be equal to the state_size
        state_size: int, dimension of U, must equal n
    Returns:
        A 2D Tensor with shape [batch x state_size] equal to arg * U
    Raises:
        ValueError if not arg.shape[1] == state_size
    """
    # assert statement here to make sure arg is a tensor etc.
    if not arg.get_shape().as_list()[1] == state_size:
        raise ValueError("Unitary expects shape[1] of first argument to be state size.")

    # TODO: better initializer for lambdas
    lie_algebra_dim = state_size*state_size

    # testing/dev options...
    INCREMENTAL_BUILD = True

    with vs.variable_scope(scope or "Unitary"):
        lambdas = vs.get_variable("Lambdas", dtype=tf.float32, shape=[1, lie_algebra_dim],
                                  initializer=tf.random_normal_initializer())
        # so like, big problem here is that we need n^2 basis elements
        # ... each of which are n^2 in size (although sparse with <=2 non-zero entries in the simple case)
        # ... that's a lot of ns!
        # solution ideas:
        #   (L is the element of the Lie algebra)
        #   (U is the element of the Lie group)
        if INCREMENTAL_BUILD:
        #   - incrementally build L using sparse (or even non-sparse) matrices
            L = lie_algebra_element(state_size, lambdas)
        elif SMALL_HIDDEN:
            raise NotImplementedError
        # initialiser is off
        #   - magically hope we don't need a large hidden state
            basis = vs.get_variable("Basis", dtype=tf.complex64,
                                    initializer=lie_algebra_basis(state_size), trainable=False)
            complex_lambdas = tf.complex(lambdas, 0)
            # TODO: make this work, np.tensordot etc...
            L = tf.matmul(complex_lambdas, basis)
        elif EXPLICIT_CONSTRUCTION:
        #   - explicitly construct L with lambdas somehow
            raise NotImplementedError
        elif SPARSE_TENSORDOT:
        #   - magically make ~tensordot work for sparse matrices
            raise NotImplementedError
        elif EXPLICIT_EXPM:
        #   - combine this solution with expm and use explicit representation of U
            raise NotImplementedError
        else:
        #   - ?????
            raise NotImplementedError
        U = L   # FOR NOW, TODO FIX, INCORRECT
        # TODO: bring this back (e.g. implement expm... possibly for sparse matrices?)
#        U = expm(tf.matmul(lambdas, basis))
        res = tf.matmul(arg, U)
    return res
