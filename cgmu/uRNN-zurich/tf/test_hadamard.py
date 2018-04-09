#!/usr/bin/env ipython
# Testing formulae for calculating derivatives of eigenvalues and eigenvectors
# of a Hermitian matrix (with simple spectrum?) according to equation 1.72 in
# https://terrytao.files.wordpress.com/2011/02/matrix-book.pdf (for e-vals) and
# and equation of my own design (for e-vecs).

import numpy as np
from scipy.linalg import expm, eigh
from unitary_np import lie_algebra_element, lie_algebra_basis_element
import pdb

# === initialise === #
n = 2
params = np.random.normal(size=n*n)
L = lie_algebra_element(n, params)
A = L
U = expm(L)

# === get eigenvalues and eigenvectors === #
# volumns of V are eigenvectors
w, V = eigh(1j*L)
w = -1j*w

# === pick a direction to take derivative in === #
for e in xrange(n*n):
    print e
#e = np.random.choice(n*n)
    # this is A dot
    Te = lie_algebra_basis_element(n, e, complex_out=True)
    dA = Te

    # === finite differences (perturb L) === #
    EPSILON = 1e-7
    L_p = L + EPSILON*Te
    U_p = expm(L_p)
    # perturbed e vals and e-vecs
    w_p, V_p = eigh(1j*L_p)
    w_p = -1j*w_p

    V_p_mod = np.zeros_like(V_p)
    # sometimes the sign of the eigenvector gets switched
    for i in xrange(n):
        u_p = V_p[:, i]
        u = V[:, i]
        if np.mean(np.abs(u_p + u)) < EPSILON:
            # sign switched
            V_p_mod[:, i] = -u_p
            print 'switching sign!', n, e
        else:
            V_p_mod[:, i] = u_p
    
    # numerical gradients 
    dw_num = (w_p - w)/EPSILON
    dV_num = (V_p_mod - V)/EPSILON
    dU_num = (U_p - U)/EPSILON

    # === analytically calculate the gradients
    dw_ana = np.zeros(shape=n, dtype=np.complex)
    dV_ana = np.zeros(shape=(n, n), dtype=np.complex)
    dV_ran = np.zeros(shape=(n, n), dtype=np.complex)
    for i in xrange(n):
        # there are n eigenvectors and eigenvalues
        u = V[:, i]
        lam = w[i]
        u_star = np.conj(u)
        # change in e-val
        dw_ana[i] = np.dot(u_star, np.dot(dA, u))
        # change in e-vec is harder
        #udot, res, rank, s = np.linalg.lstsq(L - lam*np.eye(n), np.dot(np.eye(n)*dw_ana[i] - dA, u))
        udot = np.linalg.solve(L - lam*np.eye(n), np.dot(np.eye(n)*dw_ana[i] - dA, u))
#        udot, res, rank, s = np.linalg.lstsq(lam*np.eye(n) - L, np.dot(dA - dw_ana[i]*np.eye(n), u))
        #udot, res, rank, s = np.linalg.lstsq(L_p - np.eye(n)*(dw_num[i] + lam), np.dot(-dA + np.eye(n)*dw_num[i], u))
        #udot2 = np.dot(np.dot(np.linalg.inv(lam*np.eye(n) - L), dA - dw_ana[i]), u)
        
        dV_ana[:, i] = udot
#        dV_ana[:, i] = np.dot(np.dot(np.linalg.inv(lam*np.eye(n) - L), dA - dw_ana[i]), u)
        # pick a random perpendicular vector
        dV_ran[:, i] = np.cross(u, np.random.normal(size=n))

    # now we can compare...
    print 'e-vals diff\t\t', np.mean(np.abs(dw_ana - dw_num))
    print 'e-vec diff\t\t', np.mean(np.abs(dV_ana - dV_num)), np.mean(np.abs(dV_num))
    print 'using random orthogonal vector\t', np.mean(np.abs(dV_ran - dV_num))

    # dU
    term1 = np.dot(np.dot(dV_num, np.diag(np.exp(w))), np.conj(V.T))
    term2 = np.dot(np.dot(V, np.diag(dw_ana)*np.diag(np.exp(w))), np.conj(V.T))
    term3 = np.dot(np.dot(V, np.diag(np.exp(w))), np.conj(dV_num.T))
    dU_num2 = term1 + term2 + term3
    print np.mean(np.abs(dU_num2)), ':', np.mean(np.abs(term1)), np.mean(np.abs(term2)), np.mean(np.abs(term3))

    print 'dU:\t\t', np.mean(np.abs(dU_num2 - dU_num)), np.mean(np.abs(dU_num))
