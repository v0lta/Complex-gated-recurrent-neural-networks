#!/usr/bin/env ipython
# -*- coding: utf-8 -*-
#
# For the Experiment class.
#
# ------------------------------------------
# author:       Stephanie Hyland (@corcra)
# date:         17/5/16
# ------------------------------------------

import numpy as np
from unitary_np import unitary_matrix, complex_reflection
from scipy.fftpack import fft, ifft
from functools import partial
from copy import deepcopy

# === loss functions === #
def trivial_loss(parameters, batch):
    """
    For testing.
    Parameters is just a vector, which we add to x to get y-hat. Very simple.
    """
    x, y = batch

    y_hat = x + parameters
    differences = y_hat - y
    loss = np.mean(np.square(np.linalg.norm(y_hat - y, axis=1)))
    return loss

def free_matrix_loss(parameters, batch):
    """
    For testing.
    Parameters is now a matrix!
    """
    x, y = batch
    d = x.shape[1]

    M = parameters.reshape(d, d)

    # TODO: possibly need a .T here, ack
    y_hat = np.dot(x, M.T)

    differences = y_hat - y
    loss = np.mean(np.square(np.linalg.norm(y_hat - y, axis=1)))
    return loss

def hazan_loss(parameters, batch):
    """
    The predictions are done somehow randomly, according to step 5 in 
    Algorithm 1 here: https://users.soe.ucsc.edu/~manfred/pubs/J67.pdf
    """
    x, y = batch
    n_batch, d = x.shape

    M = parameters.reshape(d, d)
    
    # actually z^T
    z = np.dot(x, M.T)
    z_lens = np.linalg.norm(z, axis=1)
    z_tilde = z/z_lens.reshape(-1, 1)
    pos_probs = (1.0 + z_lens)/2
    # if z_len > 1, this is just >1, so it's fine
    draws = np.random.random(n_batch)
    signs = (draws < pos_probs)*2 - 1
    y_hat = signs.reshape(-1, 1)*z_tilde

    differences = y_hat - y
    loss = np.mean(np.square(np.linalg.norm(y_hat - y, axis=1)))
    return loss

# === parametrisation-specific functions === #

def complex_RNN_loss(parameters, batch, permutation, theano_reflection=False):
    """
    Transform data according to the complex_RNN transformations.
    (requires importing a bunch of things and weird tensorflow hax)
    NOTE: no longer folding in any input data...

    Parameters, once again, numpy array of values.
    """
    x, y = batch
    d = x.shape[1]

    # === expand the parameters === #

    # diag1
    thetas1 = parameters[0:d]
    diag1 = np.diag(np.cos(thetas1) + 1j*np.sin(thetas1))
    # reflection 1
    reflection1_re = parameters[d:2*d]
    reflection1_im = parameters[2*d:3*d]
    # fixed permutation (get from inputs)
    # diag 2
    thetas2 = parameters[3*d:4*d]
    diag2 = np.diag(np.cos(thetas2) + 1j*np.sin(thetas2))
    # reflection 2
    reflection2_re = parameters[4*d:5*d]
    reflection2_im = parameters[5*d:6*d]
    # diag 3
    thetas3 = parameters[6*d:7*d]
    diag3 = np.diag(np.cos(thetas3) + 1j*np.sin(thetas3))

    # === do the transformation === #
    step1 = np.dot(x, diag1)
    step2 = fft(step1)/np.sqrt(d)
    step3 = complex_reflection(step2, reflection1_re, reflection1_im, theano_reflection)
    #step3 = step2
    step4 = np.dot(step3, permutation)
    step5 = np.dot(step4, diag2)
    step6 = np.sqrt(d)*ifft(step5)
    step7 = complex_reflection(step6, reflection2_re, reflection2_im, theano_reflection)
    #step7 = step6
    step8 = np.dot(step7, diag3)
    # POSSIBLY do relu_mod...

    # === now calculate the loss ... === #
    y_hat = step8
    differences = y_hat - y
    loss = np.mean(np.square(np.linalg.norm(y_hat - y, axis=1)))
    return loss

def complex_RNN_multiloss(parameters, permutations, batch):
    """
    Transform data according to the complex_RNN transformations.
    ... but now with n*n parameters.
    We achieve this by repeating the basic components.
    
    Components:
        D   (n params)
        R   (2n params)
        F   0 params
        Pi  0 params

    Canonical form:
        U = D_3 R_2 \mathcal{F}^{-1} D_2 \Pi R_1 \mathcal{F} D_1

    ... bleh, this has no obvious pattern.

    Putting two Ds beside each other is largely uninteresting.
    """
    raise NotImplementedError
    # === split up parameters, permutations

    # === combine in components

    # === get the loss

    return loss
    
    

def general_unitary_loss(parameters, batch, basis_change=None, real=False,
                         return_gradient=False):
    """
    Hey, it's my one! Rendered very simple by existence of helper functions. :)
    """
    x, y = batch
    d = x.shape[1]
    batch_size = x.shape[0]

    lambdas = parameters
    U = unitary_matrix(d, lambdas=lambdas, basis_change=basis_change, real=real)

    y_hat = np.dot(x, U.T)
    differences = y_hat - y
    loss = np.mean(np.square(np.linalg.norm(y_hat - y, axis=1)))
    if not return_gradient:
        return loss
    else:
        d_xs = np.einsum('ij, ik', differences, np.conj(x))
        ds_x = np.einsum('ij, ik', np.conj(differences), x) 
        dloss_dUre = 1.0/batch_size * (d_xs + ds_x)
        dloss_dUim = 1j*1.0/batch_size * (-d_xs + ds_x)
        return loss, dloss_dUre, dloss_dUim

# === experiment class === #
class Experiment(object):
    """
    Defines an experimental setting.
    """
    def __init__(self, name, d, 
                 project='', 
                 random_projections=0,
                 restrict_parameters=0,
                 theano_reflection=False,
                 change_of_basis=0,
                 real=False):
        # required
        self.name = name
        self.d = d
        # with defaults
        self.project = project
        self.random_projections = random_projections
        self.restrict_parameters = restrict_parameters
        self.theano_reflection = theano_reflection
        self.change_of_basis = change_of_basis
        self.real = real
        # check
        self.check_attributes()
        # defaults
        self.test_loss = -9999
        self.learning_rate = 0.001
        # derived
        self.set_basis_change()     # this must happen before set_loss...
        self.set_loss()
        self.set_learnable_parameters()
        # TODO (sparse)
        self.nonzero_index = None

    def check_attributes(self):
        """
        Make sure attributes are sensible.
        """
        if self.restrict_parameters:
            if self.d <= self.restrict_parameters:
                print 'WARNING: d is less than the factor for restricting parameters. Ignoring this setting.'
                self.restrict_parameters = False
            if not 'general' in self.name:
                print 'WARNING: restrict_parameters is only implemented for general unitary/orthogonal experiments. Setting false.'
                self.restrict_parameters = False

        if self.theano_reflection and not self.name == 'complex_RNN_vanilla':
            raise ValueError(self.name, self.theano_reflection)

        if 'projection' in self.name and not self.project:
            raise ValueError(self.name, self.project)

        if self.real:
            if 'complex_RNN' in self.name:
                raise ValueError(self.name, self.real)
            if 'unitary' in self.name:
                raise ValueError(self.name, self.real)

        if 'orthogonal' in self.name and not self.real:
            raise ValueError(self.name, self.real)

        if self.change_of_basis > 0 and not 'general' in self.name:
            raise ValueError(self.name, self.change_of_basis)
        
    def initial_parameters(self):
        """
        Return initial parameters for a given experimental setup.
        """
        d = self.d
        if self.name in {'trivial'}:
            if self.real:
                ip = np.random.normal(size=d)
            else:
                ip = np.random.normal(size=d) + 1j*np.random.normal(size=d)
        elif 'projection' in self.name or self.name == 'free_matrix':
            if self.real:
                ip = np.random.normal(size=d*d)
            else:
                ip = np.random.normal(size=d*d) + 1j*np.random.normal(size=d*d)
        elif 'complex_RNN' in self.name:
            ip = np.random.normal(size=7*d)
        elif 'general_unitary' in self.name:
            ip = np.random.normal(size=d*d)
        elif 'general_orthogonal' in self.name:
            ip = np.random.normal(size=d*(d-1)/2)
        elif 'hazan' in self.name:
            if self.real:
                ip = np.zeros(shape=(d*d))
            else:
                ip = np.zeros(shape=(d*d)) + 1j*np.zeros(shape=(d*d))
        else:
            raise ValueError(self.name)
       
        n_parameters = np.prod(ip.shape)
        assert n_parameters == self.n_parameters
        if ip.dtype == 'complex':
            assert not self.real
            n_parameters = 2*n_parameters
        print 'Initialising', n_parameters, 'real parameters.'
        return ip

    def set_loss(self):
        """
        Pick the loss function.
        """
        print '(experiment ' + self.name +'): (re)setting loss function.'
        if self.name in {'trivial'}:
            fn = trivial_loss
            self.n_parameters = self.d
        elif 'projection' in self.name or self.name == 'free_matrix':
            fn = free_matrix_loss
            self.n_parameters = self.d*self.d
        elif 'hazan' in self.name:
            fn = hazan_loss
            self.n_parameters = self.d*self.d
        elif 'complex_RNN' in self.name:
            permutation = np.random.permutation(np.eye(self.d))
            fn = partial(complex_RNN_loss, permutation=permutation, 
                         theano_reflection=self.theano_reflection)
            self.n_parameters = 7*self.d
        elif 'general_' in self.name:
            if self.change_of_basis > 0 and self.basis_change is None:
                self.set_basis_change()
            fn = partial(general_unitary_loss, 
                         basis_change=self.basis_change,
                         real=self.real)
            if 'unitary' in self.name:
                self.n_parameters = self.d*self.d
            elif 'orthogonal' in self.name:
                self.n_parameters = self.d*(self.d-1)/2
            else:
                raise ValueError(self.name)
        else:
            raise ValueError(self.name)

        self.loss_function = fn
        return True

    def set_learnable_parameters(self):
        d = self.d
        if self.restrict_parameters:
            learnable_parameters = np.random.choice(d*d, self.restrict_parameters*d, replace=False)
            self.learnable_parameters = learnable_parameters
        else:
            self.learnable_parameters = np.arange(self.n_parameters)
        return True

    def set_basis_change(self):
        if self.change_of_basis > 0:
            d = self.d
            scale = self.change_of_basis
            basis_change = np.random.uniform(low=-scale, high=scale, size=(d*d,d*d))
            self.basis_change = basis_change
        else:
            self.basis_change = None
        return True

# === specific experimental designs === #
def presets(d):
    """
    Returns a list of 'preset' experiment objects.
    """
    proj = Experiment('projection', d, project='polar')
    complex_RNN = Experiment('complex_RNN', d)
    general = Experiment('general_unitary', d)
    exp_list = [proj, complex_RNN, general]
    if d > 7:
        general_restrict = Experiment('general_unitary_restricted', d, restrict_parameters=7)
        exp_list.append(general_restrict)
    return exp_list

def test_random_projections(d=6):
    exp_list = []
    if d == 6:
        #for j in np.linspace(np.sqrt(d), 0.5*d*(d-1), num=5, dtype=int):
        for j in [4, 9, 16, 25, 36]:
            exp_list.append(Experiment('general_unitary_' + str(j), d, random_projections=j))
    elif d == 12:
        for j in [9, 25, 49, 81, 144]:
            exp_list.append(Experiment('general_unitary_' + str(j), d, random_projections=j))
    exp_list.append(Experiment('general_unitary', d))
    return exp_list

def basis_change(d):
    """ testing how the change of basis influences learning """
    general_default = Experiment('general_unitary', d)
    general_basis_1 = Experiment('general_unitary_basis10', d, change_of_basis=10)
    general_basis_2 = Experiment('general_unitary_basis50', d, change_of_basis=50)
    exp_list = [general_default, general_basis_1, general_basis_2]
    return exp_list

def rerun(d):
    """
    Steph is silly.
    """
    proj = Experiment('projection', d, project='polar')
    complex_RNN = Experiment('complex_RNN', d)
    general = Experiment('general_unitary', d)
#    general_basis_5 = Experiment('general_unitary_basis5', d, change_of_basis=5)
    #exp_list = [proj, complex_RNN, general, general_basis_5]
    exp_list = [proj, complex_RNN, general]
    if d > 7:
        general_restricted = Experiment('general_unitary_restricted', d, restrict_parameters=7)
        exp_list.append(general_restricted)
    return exp_list

def basis_test(d):
    """
        self.learning_rate = 0.001
        and testing with learning rate...
    """
    initial_lr = 1e-3 ## initial lr
    exp_list = []
    bases = [5.0, 10.0, 20.0]
    lrs = [initial_lr] + [initial_lr/((x*x)/2) for x in bases]
    print lrs
    #lr_adj = lr/(basis_change/2)
    for basis_change in bases:
        print basis_change
        for lr in lrs:
            print lr
            gen_nobasis = Experiment('general_unitary_lr'+str(lr), d)
            gen_basis = Experiment('general_unitary_basis' + str(basis_change)+ '_lr'+str(lr), d, change_of_basis=basis_change)
            gen_nobasis.learning_rate = lr
            gen_basis.learning_rate = lr
            exp_list.append(gen_nobasis)
            exp_list.append(gen_basis)
    return exp_list

# === more experiments 1/6/16 === #
def test_orth(d):
    """ testing new features """
    proj_real = Experiment('projection_orthogonal', d, project='polar', real=True)
    proj_complex = Experiment('projection_unitary', d, project='polar')
    general_unitary = Experiment('general_unitary', d)
    general_orthogonal = Experiment('general_orthogonal', d, real=True)
    exp_list = [proj_real, proj_complex, general_unitary, general_orthogonal]
    return exp_list

# === hazan === #
def hazan(d):
    lr = 2e-7
    h_real = Experiment('hazan_real', d, project=False, real=True)
#    h_imag = Experiment('hazan_imag', d, project=False, real=False)
    general_unitary = Experiment('general_unitary', d)
    complex_RNN = Experiment('complex_RNN', d)
    proj_real = Experiment('projection_orthogonal', d, project='polar', real=True)
#    general_orthogonal = Experiment('general_orthogonal', d, real=True)
    exp_list = [h_real, general_unitary, proj_real]
    for e in exp_list:
        if 'hazan' in e.name:
            e.learning_rate = lr
    return exp_list

# === compare projections === #
def project_compare(d):
    proj_polar = Experiment('projection_polar', d, project='polar')
    proj_evals = Experiment('projection_evals', d, project='evals')
    complex_RNN = Experiment('complex_RNN', d)
    general = Experiment('general_unitary', d)
    exp_list = [proj_polar, proj_evals, complex_RNN, general]
    if d > 7:
        general_restricted = Experiment('general_unitary_restricted', d, restrict_parameters=7)
        exp_list.append(general_restricted)
    return exp_list
