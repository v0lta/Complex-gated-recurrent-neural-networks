#!/usr/bin/env ipython
#
# Script for checking possible gradient solutions.
#
#
# ------------------------------------------
# author:       Stephanie Hyland (@corcra)
# date:         26/5/16
# ------------------------------------------
#
#

import numpy as np
from scipy.linalg import expm, eigh
from unitary_np import lie_algebra_element, lie_algebra_basis
import unitary_np
import pdb

def eig_trick(thetas, vectorised=True, intermediate=False):
    n = int(np.sqrt(len(thetas)))
    L = lie_algebra_element(n, thetas)
    w, v = eigh(1j*L)
    w = -1j*w
    vdag = np.conj(v.T)
    # w are the e-vals, v has columns for eigenvectors
    expw = np.exp(w)
    Ts = lie_algebra_basis(n)
    # going to do this the SLOWEST WAY
    one = np.ones([n, n])
    if not vectorised:
        grads = np.empty(shape=(n*n, n, n), dtype=complex)
        for (i, T) in enumerate(Ts):
            G = np.dot(np.dot(vdag, T), v)
            V = np.empty(shape=(n, n), dtype=complex)
            # intermediate speedup maybe
            if intermediate:
                V = ((expw*one - (expw*one).T)/(w*one - (w*one).T))*G 
                V[xrange(n), xrange(n)] = expw*G[xrange(n), xrange(n)]
            else:
                for r in xrange(n):
                    V[r, r] = G[r, r]*expw[r]
                    for s in xrange(r + 1, n):
                        V[r, s] = (expw[r] - expw[s])*G[r, s]/(w[r] - w[s])
                        V[s, r] = (expw[s] - expw[r])*G[s, r]/(w[s] - w[r])
            grad = np.dot(np.dot(v, V), vdag)
            grads[i, :, :] = grad
    else:
        # TODO: make faster
        G = np.einsum('ij,ajk,kl', vdag, Ts, v)
        one = np.ones([n, n])
        # offdiag
        V = ((expw*one - (expw*one).T)/(w*one - (w*one).T))*G
        # diag
        V[:, xrange(n), xrange(n)] = expw*G[:, xrange(n), xrange(n)]
        # now the grads
        grads = np.einsum('ij, ajk, kl', v, V, vdag)
    # this will give NaNs along the diagonal, oh well
#    G = ((expw*one - (expw*one).T)/(w*one - (w*one).T))*Ts
    # now we deal with the diags
#    G[:, xrange(n), xrange(n)] = expw*Ts[:, xrange(n), xrange(n)]
    # now for the grads
#    grads = np.einsum('jk,ikl', np.conj(v.T), np.dot(G, v))
    return grads

def finite_difference(thetas, EPSILON=1e-6):
    n = int(np.sqrt(len(thetas)))
    L = lie_algebra_element(n, thetas)
    U = expm(L)
    Ts = lie_algebra_basis(n)
    grads = np.empty(shape=(n*n, n, n), dtype=complex)
    for (i, T) in enumerate(Ts):
        U_perturb = expm(L + EPSILON*T)
        grad = (U_perturb - U)/EPSILON
        grads[i, :, :] = grad
    return grads
