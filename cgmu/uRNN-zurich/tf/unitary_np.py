#!/usr/bin/env ipython
# -*- coding: utf-8 -*-
#
# Functions pertaining to the unitary group and its associated Lie algebra.
# (RETURN NP OBJECTS)
#
# ------------------------------------------
# author:       Stephanie Hyland (@corcra)
# date:         12/4/16
# ------------------------------------------

import numpy as np
import pdb

from scipy.linalg import expm, polar, eigh, eig
from scipy.fftpack import fft, ifft

from functools import partial

# === generic unitary === #
def complex_reflection(x, v_re, v_im, theano_reflection=False):
    """
    Hey, it's this function again! Woo!
    NOTE/WARNING: theano_reflection gives a DIFFERENT RESULT to the other one...
    see this unresolved issue:
    https://github.com/amarshah/complex_RNN/issues/2
    """
    if theano_reflection:
        # (mostly copypasta from theano, with T replaced by np all over)
        # FOR NOW OK
        input_re = np.real(x)
        # alpha
        input_im = np.imag(x)
        # beta
        reflect_re = v_re
        # mu
        reflect_im = v_im
        # nu

        vstarv = (reflect_re**2 + reflect_im**2).sum()

        # (the following things are roughly scalars)
        # (they actually are as long as the batch size, e.g. input[0])
        input_re_reflect_re = np.dot(input_re, reflect_re)
        # αμ
        input_re_reflect_im = np.dot(input_re, reflect_im)
        # αν
        input_im_reflect_re = np.dot(input_im, reflect_re)
        # βμ
        input_im_reflect_im = np.dot(input_im, reflect_im)
        # βν

        a = np.outer(input_re_reflect_re - input_im_reflect_im, reflect_re)
        # outer(αμ - βν, mu)
        b = np.outer(input_re_reflect_im + input_im_reflect_re, reflect_im)
        # outer(αν + βμ, nu)
        c = np.outer(input_re_reflect_re - input_im_reflect_im, reflect_im)
        # outer(αμ - βν, nu)
        d = np.outer(input_re_reflect_im + input_im_reflect_re, reflect_re)
        # outer(αν + βμ, mu)

        output_re = input_re - 2. / vstarv * (d - c)
        output_im = input_im - 2. / vstarv * (a + b)

        output = output_re + 1j*output_im
    else:
        # do it the 'old fashioned' way
        v = v_re + 1j*v_im
        # aka https://en.wikipedia.org/wiki/Reflection_%28mathematics%29#Reflection_through_a_hyperplane_in_n_dimensions
        # but with conj v dot with x
        output = x - (2.0/np.dot(v, np.conj(v))) * np.outer(np.dot(x, np.conj(v)), v)

    return output

# === lie algebra === #
def lie_algebra_element(n, lambdas, check_skew_hermitian=False, check_real=False):
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
        lambdas:        a ndarray of length n x n
        check_skew_hermitian
    Returns:
        a rank-2 numpy ndarray dtype complex, the element of the Lie algebra

    POSSIBLE TODO: create basis-element generator
    """
    if len(lambdas) == n*(n-1)/2:
        # orthoognal case (subspace of the lie algebra!)
        L = np.zeros(shape=(n, n))
        lambda_index = 0
        for i in xrange(0, n):
            for j in xrange(0, i):
                # the i > j cases are the real ones
                e = n*i + j
                T_re, T_im = lie_algebra_basis_element(n, e, complex_out=False)
                if check_real:
                    # the T_im *should* be zero
                    assert np.array_equal(T_im, np.zeros_like(T_im))
                L += lambdas[lambda_index]*T_re
                lambda_index += 1
    elif len(lambdas) == n*n:
        # unitary case
        L = np.zeros(shape=(n, n), dtype=complex)
        for e in xrange(0, n*n):
            T = lie_algebra_basis_element(n, e, complex_out=True)
            L += lambdas[e]*T
    else:
        raise ValueError(lambdas)

    if check_skew_hermitian:
        assert np.array_equal(np.transpose(np.conj(L)), -L)

    return L

def lie_algebra_basis_element(n, e, check_skew_hermitian=False, complex_out=False):
    """
    Return a *single* element (the e-th one) of a basis of u(n).
    See lie_algebra_basis for more details.

    Args:
        n:  as in u(n)
        e:  which element?

    Returns:
        Two numpy arrays: the real part and imaginary parts of the element.
    """
    lie_algebra_dim = n*n
    T_re = np.zeros(shape=(n, n), dtype=np.float32)
    T_im = np.zeros(shape=(n, n), dtype=np.float32)

    # three cases: imaginary diagonal, imaginary off-diagonal, real off-diagonal
    # (reasonably sure this is how you convert these, it makes sense to me...)
    i = e / n
    j = e % n
    if i > j:
        # arbitrarily elect these to be the real ones
        T_re[i, j] = 1
        T_re[j, i] = -1
    else:
        # (includes i == j, the diagonal part)
        T_im[i, j] = 1
        T_im[j, i] = 1
    if check_skew_hermitian:
        basis_matrix = T_re + 1j*T_im
        assert np.array_equal(np.transpose(np.conjugate(basis_matrix)), -basis_matrix)
    if complex_out:
        return T_re + 1j*T_im
    else:
        return T_re, T_im

def lie_algebra_basis(n):
    """
    Generate a basis of the Lie algebra u(n), associated to the Lie group U(n) of n x n unitary matrices.
    The Lie algebra u(n) is n x n skew-Hermitian matrices. 
    A skew-Hermitian matrix is a square complex matrix A such that A^{dagger} = -A
    This means that conj(A_ij) = -A_ji
    The Lie algebra has real dimension n^2.

    One construction for the basis is as follows:
    - the first n elements are:
        diag(..., i, ...) (i in the nth position)
        ... Since diag(...)^T = diag(...) and conj(i) = -i
    - the next n(n-1)/2 elements are:
        a = 1..n, b = a..n, A_ab = 1, A_ba = -1
        ... Since real, so we just need A^T = -A
    - the next n(n-1)/2 elements are:
        a = 1..n, b = a..n, A_ab = i, A_ba = i
        ... Since imaginary, so we just need A^T = A
    
    Note that I will construct the real and imaginary parts separatey, and join them at the end as a tf.tensor.

    Args:
        n: see above
    Returns:
        rank-3 numpy ndarray, complex, list of the n^2 basis matrices of the Lie algebra  
    
    TODO: double-check all the maths here
    """
    lie_algebra_dim = n*n
    tensor_re = np.zeros(shape=(lie_algebra_dim, n, n), dtype=np.float32)
    tensor_im = np.zeros(shape=(lie_algebra_dim, n, n), dtype=np.float32)

    for e in xrange(0, lie_algebra_dim):
        T_re, T_im = lie_algebra_basis_element(n, e)
        tensor_re[e, :, :] = T_re
        tensor_im[e, :, :] = T_im

    # ensure they are indeed skew-Hermitian
    basis = tensor_re + 1j*tensor_im 
    for basis_matrix in basis:
        assert np.array_equal(np.transpose(np.conjugate(basis_matrix)), -basis_matrix)
       
    return basis

def project_to_unitary(parameters, check_unitary=True):
    """
    Use polar decomposition to find the closest unitary matrix.
    """
    # parameters must be a square number (TODO: FIRE)
    n = int(np.sqrt(len(parameters)))

    A = parameters.reshape(n, n)
    U, p = polar(A, side='left')

    if check_unitary:
        # (np.conj is harmless for real U)
        assert np.allclose(np.dot(U, np.conj(U.T)), np.eye(n))

    parameters = U.reshape(n*n)
    return parameters

def project_with_evals(parameters):
    """
    Do an eigenvalue decomposition and fix eigenvalues absolute val 1
    """
    # parameters must be a square number (TODO: FIRE)
    n = int(np.sqrt(len(parameters)))

    A = parameters.reshape(n, n)

    w, v = eig(A)

    # project w
    w_normed = np.array([x/np.linalg.norm(x) for x in w])

    # rebuild
    A_new = np.dot(np.dot(v, np.diag(w_normed)), np.linalg.inv(v))
    
    parameters = A_new.reshape(n*n)
    return parameters

# ===
def new_basis_lambdas(lambdas, basis_change):
    """
    Updates lambdas according to change of basis.
    """
    new_lambdas = np.dot(lambdas, basis_change)
    return new_lambdas

def random_unitary_composition(n):
    """
    Returns a random matrix generated from the unitary composition procedure
    of Arjovsky et al [https://arxiv.org/abs/1511.06464]
    
    ... skipping reflection because tired TODO fix
    """
    # arjovsky things
    scale = np.sqrt(6.0/ ( 2 + n*2))
    # diag
    thetas1 = np.random.uniform(low=-np.pi, high=np.pi, size=n)
    diag1 = np.diag(np.cos(thetas1) + 1j*np.sin(thetas1))
    # fft
    step2 = fft(diag1)
    # reflection
    v1_re = np.random.uniform(low=-scale, high=scale, size=n)
    v1_im = np.random.uniform(low=-scale, high=scale, size=n)
    step3 = complex_reflection(step2, v1_re, v1_im, theano_reflection=False)
    # permutation
    permutation = np.random.permutation(np.eye(n))
    step4 = np.dot(step3, permutation)
    # diag
    thetas2 = np.random.uniform(low=-np.pi, high=np.pi, size=n)
    diag2 = np.diag(np.cos(thetas2) + 1j*np.sin(thetas2))
    step5 = np.dot(step4, diag2)
    # ifft
    step6 = ifft(step5)
    # reflection
    v2_re = np.random.uniform(low=-scale, high=scale, size=n)
    v2_im = np.random.uniform(low=-scale, high=scale, size=n)
    step7 = complex_reflection(step6, v2_re, v2_im, theano_reflection=False)
    # final diag
    thetas3 = np.random.uniform(low=-np.pi, high=np.pi, size=n)
    diag3 = np.diag(np.cos(thetas3) + 1j*np.sin(thetas3))
    U = np.dot(step7, diag3)
    return U

def unitary_matrix(n, method='lie_algebra', lambdas=None, check_unitary=True,
                   basis_change=None, real=False, special=False):
    """
    Returns a random (or not) unitary (or orthogonal) matrix of dimension n x n.
    I give no guarantees about the distribution we draw this from.
    To do it 'properly' probably requires a Haar measure.

    Note on 'method':
        - Lie algebra representation (optionally provide lambdas)
        - Using qr decomposition of a random square complex matrix
        - Something which could have been generated by the complex_RNN 
            parametrisation

    Note on 'basis_change':
        Consider the basis defined by lie_algebra_basis(_element) to be the
        'canonical'/standard basis, e.g. the coordinates [0 ... 1 ... 0] etc.
        A new basis element can be represented simply by a vector in this basis
        [alpha_1, ..., alpha_N], which is really
            sum_i alpha_i T_i
        where T_i is the basis element with coordinates [0... 1_i ... 0]
            (to abuse notation slightly)
        So in general, the jth new basis element will be given by
            V_j = sum_i alpha_{ji} T_i
        This matrix alpha_ji defines the change of basis, and is what should be
        provided for this argument.

        Now, in this code I'm using a trick of sorts. Since ultimately we output
            sum_i lambda_i T_i
        we can avoid having to _explicitly_ calculate the new basis elements by
        modifying lambda_i

        If lambda_i are assumed given _with respect to_ the basis V, then we
        wish to calculate
            L = sum_i lambda_i V_i = sum_i lambda_i sum_j alpha_{ij} T_j
              = sum_{i, j} lambda_i alpha_{ij} T_j

        If we define ~lambda_j = sum_i lambda_i alpha_{ij}, then we have
            L = sum_j ~lambda_j T_j

        So calculating L with respect to a new basis is the same as calculating
        L with respect to the old basis, but using a different set of lambdas.
        
        So, in this script when we 'change basis', we're really just changing
        lambda.

    Note on 'special':
        For the Lie algebra method:
        Special unitary/orthogonal matrices are generated by *traceless*
        skew-Hermitian/symmetric matrices. This means the parameters
        corresponding to the 'i somewhere on the diagonal' basis have to
        sum to zero. We'll do that by sampling (n-1) of them, and setting the
        last one to the negative sum of the rest.

        For other methods:
        TODO

    Args:
        n               unitary matrix is size n x n
        method          way of generating the unitary matrix (see note above)
        lambdas         (if method == lie algebra) the set of coefficients with 
                            respect to the basis of the algebra
        check_unitary   check if the result is unitary (mostly for testing)
                            before returning
        basis_change    (if method == lie algebra)  a matrix giving 
                            coefficients of 'new' basis wrt old basis (see above)
        real            return a real (= orthogonal) matrix?
        special         return a special unitary/orthogonal matrix? (det=1)

    Returns:
        U               the unitary/orthogonal matrix
    """
    # give warnings if unnecessary options provided
    if not method == 'lie_algebra':
        if not lambdas is None:
            print 'WARNING: Method', method, 'selected, but lambdas provided (uselessly)!'
        if not basis_change is None:
            print 'WARNING: Method', method, 'selected, but basis change provided (uselessly)!'
        if special:
            print 'WARNING: Asked for special unitary/orthogonal matrix, but Lie algebra method not selected. This option is not yet implemented for', method, ': ignoring!'
    # now on to generating the Us
    
    if method == 'lie_algebra':
        if lambdas is None:
            # create the lambdas
            lambdas = np.zeros(shape=n*n)
            if real:
                # all lambdas are zero except for ones corresponding to real
                real_indices = [e for e in xrange(n*n) if e/n > e % n]
                lambdas[real_indices] = np.random.normal(size=(len(real_indices)))
                # real implies special, for the lie algebra method
            else:
                if special:
                    diagonal_indices = [e for e in xrange(n*n) if e/n == e % n]
                    non_diagonal_indices = [x for x in xrange(n*n) if not x in diagonal_indices]
                    # omit one randomly
                    fixed_index = np.random.choice(diagonal_indices)
                    diagonal_values = np.random.normal(size=len(diagonal_indices)-1)
                    fixed_value = -np.sum(diagonal_values)
                    unfixed_indices = [x for x in diagonal_indices if not x == fixed_index]
                    lambdas[unfixed_indices] = diagonal_values
                    lambdas[fixed_index] = fixed_value
                    lambdas[non_diagonal_indices] = np.random.normal(len(non_diagonal_indices))
                else:
                    lambdas = np.random.normal(size=n*n)
        if not basis_change is None:
            lambdas = new_basis_lambdas(lambdas, basis_change)
        L = lie_algebra_element(n, lambdas)
        U = expm(L)
    elif method == 'qr':
        if real:
            A = np.random.random(size=(n, n))
        else:
            A = np.random.random(size=(n, n)) + 1j*np.random.random(size=(n, n))
        U, r = np.linalg.qr(A)
    elif method == 'composition':
        if real:
            raise ValueError(real)
        U = random_unitary_composition(n)
    else:
        raise ValueError(method)
    
    if check_unitary:
        assert np.allclose(np.dot(U, np.conj(U.T)), np.eye(n))
    if special or (method == 'lie_algebra' and real):
        assert np.abs(np.linalg.det(U) - 1) < 1e-6
    return U

# === some grad hacks === #

def numerical_partial_gradient(e, L, n, U, dcost_dU_re, dcost_dU_im, EPSILON):
    """ For giving to the pool.
    """
    dU_dlambda = (expm(L + EPSILON*lie_algebra_basis_element(n, e, complex_out=True)) - U)/EPSILON
    delta = np.trace(np.dot(dcost_dU_re.T, np.real(dU_dlambda)) + np.dot(dcost_dU_im.T, np.imag(dU_dlambda)))
    return delta

def numgrad_lambda_update(dcost_dU_re, dcost_dU_im, lambdas, learning_rate,
                          EPSILON=1e-5):
    """
    Given dcost/dU, get dcost/dlambdas
    Using... numerical differentiation. :(
    """
    # first term (d cost / d U)
    assert dcost_dU_re.shape == dcost_dU_im.shape
    n = dcost_dU_re.shape[0]
    assert len(lambdas) == n*n

    # second term (d U / d lambdas)
    # meanwhile update lambdas
    L = lie_algebra_element(n, lambdas)
    U = expm(L)

    # parallel version
    numerical_parallel = partial(numerical_partial_gradient, L=L, U=U, n=n, 
                                 dcost_dU_re=dcost_dU_re, dcost_dU_im=dcost_dU_im, 
                                 EPSILON=EPSILON)
    dlambdas = np.array(map(numerical_parallel, xrange(n*n)))
    lambdas += learning_rate*dlambdas
    
    # having updated the lambdas, get new U
    U_new = expm(lie_algebra_element(n, lambdas))

    return np.real(U_new), np.imag(U_new), dlambdas

def eigtrick_speedy(n, w, expw, v, vdag, lambdas, learnable_parameters=None):
    if learnable_parameters is None:
        learnable_parameters = xrange(len(lambdas))
    one = np.ones([n, n])
    M = (expw*one - (expw*one).T)/(w*one - (w*one).T)
    M[xrange(n), xrange(n)] = expw
    dU_dlambdas = np.zeros(shape=(len(lambdas), n, n), dtype=complex)
    lambda_index = 0
    for r in xrange(n):
        for s in xrange(n):
            if r > s:
                # real asymmetric case
                WTW = - np.outer(vdag[:, r], v[s, :]) + np.outer(vdag[:, s], v[r, :])
            else:
                if len(lambdas) == n*(n-1)/2:
                    # ditch it all
                    continue
                elif r == s:
                    # imag diag case
                    WTW = 1j*np.outer(vdag[:, r], v[r, :])
                else:
                    # imag symmetric
                    WTW = 1j*(np.outer(vdag[:, s], v[r, :]) + np.outer(vdag[:, r], v[s, :]))
            V = WTW*M
            dU_dlambda = np.dot(np.dot(v, V), vdag)
            if lambda_index in learnable_parameters:
                # this definitely isn't the most efficient way to do this
                dU_dlambdas[lambda_index, :] = dU_dlambda
            lambda_index += 1
    return dU_dlambdas


def eigtrick_lambda_update(dcost_dU_re, dcost_dU_im, lambdas, learning_rate, 
                         speedy=False):
    """
    Given dcost/dU, get dcost/dlambdas
    Using... linear algebra!

    NOTE: the TF implementation actually works with U transpose.

    So dcost_dU is transposed, and the thing we return shall also be transposed.
    """
    n = dcost_dU_re.shape[0]
    assert len(lambdas) == n*n or len(lambdas) == n*(n-1)/2

    L = lie_algebra_element(n, lambdas)
    dlambdas = np.zeros_like(lambdas)
    # here comes the trick
    # TEST DEBUG CHECK TODO
    w, v = eigh(1j*L.T)
    # TEST DEBUG CHECK TODO  (WHY DID THIS WORK?//?)
    w = -1j*w
    vdag = np.conj(v.T)
    expw = np.exp(w)
    if not speedy:
        Ts = lie_algebra_basis(n)
        if len(lambdas) == n*(n-1)/2:
            # orthogonal case
            T_indices = []
            # itertools probably has a better solution for this but w/e
            for i in xrange(0, n):
                for j in xrange(0, i):
                    e = n*i + j
                    T_indices.append(e)
        elif len(lambdas) == n*n:
            T_indices = xrange(n*n)

        #   so parallelisable!!!
        for (i, e) in enumerate(T_indices):
            T = Ts[e]
            G = np.dot(np.dot(vdag, T), v)
            V = np.empty(shape=(n, n), dtype=complex)
            for r in xrange(n):
                V[r, r] = G[r, r]*expw[r]
                for s in xrange(r + 1, n):
                    V[r, s] = (expw[r] - expw[s])*G[r, s]/(w[r] - w[s])
                    V[s, r] = (expw[s] - expw[r])*G[s, r]/(w[s] - w[r])
            grad = np.dot(np.dot(v, V), vdag)
            dU_dlambda = grad
            #delta = np.trace(np.dot(dcost_dU_re, np.real(dU_dlambda)) + np.dot(dcost_dU_im, np.imag(dU_dlambda)))
            #delta = np.trace(np.dot(dcost_dU_re.T, np.real(dU_dlambda)) + np.dot(dcost_dU_im.T, np.imag(dU_dlambda)))
            delta = np.trace(np.dot(dcost_dU_re, np.real(dU_dlambda)) + np.dot(dcost_dU_im, np.imag(dU_dlambda)))
            dlambdas[i] = delta
    else:
        # woooo
        one = np.ones([n, n])
        M = (expw*one - (expw*one).T)/(w*one - (w*one).T)
        M[xrange(n), xrange(n)] = expw
        # k
        lambda_index = 0
        # damn a n^2 calculation? gross
        for r in xrange(n):
            for s in xrange(n):
                if r > s:
                    # real asymmetric case
                    WTW = - np.outer(vdag[:, r], v[s, :]) + np.outer(vdag[:, s], v[r, :])
                else:
                    if len(lambdas) == n*(n-1)/2:
                        # ditch it all
                        continue
                    elif r == s:
                        # imag diag case
                        WTW = 1j*np.outer(vdag[:, r], v[r, :])
                    else:
                        # imag symmetric
                        WTW = 1j*(np.outer(vdag[:, s], v[r, :]) + np.outer(vdag[:, r], v[s, :]))
                V = WTW*M
                dU_dlambda = np.dot(np.dot(v, V), vdag)
#                delta = np.trace(np.dot(dcost_dU_re, np.real(dU_dlambda)) + np.dot(dcost_dU_im, np.imag(dU_dlambda)))
                delta = np.trace(np.dot(dcost_dU_re.T, np.real(dU_dlambda)) + np.dot(dcost_dU_im.T, np.imag(dU_dlambda)))
                dlambdas[lambda_index] = delta
                lambda_index += 1

    # done-ish, recombine
    lambdas -= learning_rate*dlambdas
   
    # having updated the lambdas, get new U
    #U_new = expm(lie_algebra_element(n, lambdas))
    U_new = expm(lie_algebra_element(n, lambdas)).T
    return np.real(U_new), np.imag(U_new), dlambdas
