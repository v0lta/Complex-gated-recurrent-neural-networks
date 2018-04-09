#!/usr/bin/env ipython
#
#
# Scripts to help me find a good approximation to the gradient of exp(L).
# Specifically:
#   Script to test when/if I can truncate the infinite sum for calculating the gradient.
#       aka maths is hard, code is easy
#   Script to check difference between gradients obtained numerically and using
#       whatever approximation...
#
# ------------------------------------------
# author:       Stephanie Hyland (@corcra)
# date:         6/5/16
# ------------------------------------------
#

import numpy as np
from scipy.misc import factorial
from scipy.linalg import expm
import unitary
import pdb

EPSILON=1e-5
# === wat === #
# (i am doing myself a regret)
DO_LTL_NORMS = False
DO_UT_TU_HACK = True

# === relating to the LTL sum thing === #
# == CONSTANTS == #
J_MAX = 100
N_REPS = 20

# == SETUP == #
def get_L_and_Ts(n):
    # (I have irritatingly written a random set of the unitary functions for tf)
    # get the the basis set (so inefficient!)
    Ts = np.zeros(shape=(n*n, n, n), dtype=complex)
    for e in xrange(n*n):
        T_re, T_im = unitary.lie_algebra_basis_element(n, e)
        Ts[e, :, :] = T_re + 1j*T_im

    # create L by choosing random lambdas
    lambdas = np.random.normal(size=(n*n))
    L = np.einsum('i,ijk', lambdas, Ts)

    return L, Ts

# == the accumulator on its own == #
def get_LTL(L, T, J):
    # (so inefficient)
    n = L.shape[0]
    # trace will store np.mean(L^{j-a} T L^a)/j!
    trace = [0]*J_MAX
    # generate powers of L as we go along
    L_powers = np.zeros(shape=(J_MAX, n, n), dtype=complex)
    L_powers[0] = np.eye(n)
    trace[0] = 1
    #trace[1] = np.linalg.norm(np.mean(np.dot(L, T)))
    trace[1] = trace[0] + np.linalg.norm(np.mean(np.dot(L, L)))
    L_powers[1] = L
    for j in xrange(2, J):
        L_powers[j] = np.dot(L, L_powers[j-1])
    # now get the accumulator
    LTL = np.zeros_like(L)
    for a in xrange(J):
        LTL += np.dot(np.dot(L_powers[J-a], L), L_powers[a])
    LTL = LTL/factorial(J)
    return LTL

# == GET TRACE == #
def get_LTL_norm_trace(L, T, J_MAX):
    # (so inefficient)
    n = L.shape[0]
    # trace will store np.mean(L^{j-a} T L^a)/j!
    trace = [0]*J_MAX
    # generate powers of L as we go along
    L_powers = np.zeros(shape=(J_MAX, n, n), dtype=complex)
    L_powers[0] = np.eye(n)
    trace[0] = 1
    #trace[1] = np.linalg.norm(np.mean(np.dot(L, T)))
    trace[1] = trace[0] + np.linalg.norm(np.mean(np.dot(L, L)))
    L_powers[1] = L
    for j in xrange(2, J_MAX):
        L_powers[j] = np.dot(L, L_powers[j-1])
        accumulator = np.zeros_like(L)
        for a in xrange(j):
            #accumulator += np.dot(np.dot(L_powers[j-a], T), L_powers[a])
            accumulator += np.dot(np.dot(L_powers[j-a], L), L_powers[a])
        accumulator /= factorial(j)
#        pdb.set_trace()
        trace_complex = np.mean(accumulator)
        trace[j] = trace[j-1] + np.linalg.norm(trace_complex)
        if trace[j] > 100:
            pdb.set_trace()
    return trace

# == GET COMBOTRACE == #
def get_multiple_LTL_norm_traces(N, N_REPS, f=None):
    combo_trace = np.zeros(shape=(J_MAX, N_REPS))
    for l in xrange(N_REPS):
        L, Ts = get_L_and_Ts(N)
        i = np.random.choice(n*n)
        T = Ts[i, :, :]
        trace = get_LTL_norm_trace(L, T, J_MAX)
        assert len(trace) == J_MAX
        if f is not None:
            for (j, t) in enumerate(trace):
                f.write(str(j) + ' ' + str(N)+ ' ' + str(l) + ' ' + str(t) + '\n')
        combo_trace[:, l] = trace
    if f is None:
        means = np.mean(combo_trace, axis=1)
        stds = np.std(combo_trace, axis=1)
        results = zip(list(means), list(stds))
        for m, s in results:
            print m, s
    return combo_trace

# === MAIN === #
if DO_LTL_NORMS:
    fout = open('gradient_test_traces.txt', 'w')
    fout.write('j d rep val\n')
    for n in [1, 2, 5, 10, 15, 25, 50]:
        print n
        ct = get_multiple_LTL_norm_traces(n, N_REPS, fout)
    fout.close()

if DO_UT_TU_HACK:
    n = 5
    J = 13
    L, Ts = get_L_and_Ts(n)
    U = expm(L)
#    hack_gradients = np.array([np.dot(U, T) + np.dot(T, U) for T in Ts])
#    hack_gradients = np.array([get_LTL(L, T, J) for T in Ts])
    hack_gradients = np.array([np.dot(L, T) - np.dot(T, L) for T in Ts])
    numerical_gradients = np.array([(expm(L + EPSILON*T) - expm(L))/EPSILON for T in Ts])
    differences = hack_gradients - numerical_gradients
    pdb.set_trace()
    print 'hacks'
    print np.linalg.norm(hack_gradients, ord='fro', axis=(1, 2))
    print 'numerical'
    print np.linalg.norm(numerical_gradients, ord='fro', axis=(1, 2))
    print 'differences'
    print np.linalg.norm(differences, ord='fro', axis=(1, 2))
    pdb.set_trace()
