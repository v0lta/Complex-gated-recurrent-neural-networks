# pylint: disable=E722

# Recreation of the Montreal adding problem experiments from Arjovski et al.
# Working with Tensorflow 1.3
import pickle
import tensorflow as tf
import argparse
import custom_cells as cc
import numpy as np
import scipy.stats as scistats

# import state to state non-linearities
from custom_cells import mod_relu
from custom_cells import hirose
from custom_cells import linear
from custom_cells import moebius
from custom_cells import relu
from custom_cells import split_relu
from custom_cells import z_relu
from custom_cells import tanh

# import gate non-linearities
from custom_cells import gate_phase_hirose
from custom_cells import mod_sigmoid_prod
from custom_cells import mod_sigmoid_sum
from custom_cells import mod_sigmoid
from custom_cells import mod_sigmoid_beta
from custom_cells import real_mod_sigmoid_beta


from synthetic_experiments import main
from IPython.core.debugger import Tracer

debug_here = Tracer()

if __name__ == "__main__":
    # time_steps=100, n_train=int(2e6), n_test=int(1e4),
    # n_units=512, learning_rate=1e-3, decay=0.9,
    # batch_size=50, GPU=0, memory=False, adding=True,
    # cell_fun=tf.contrib.rnn.LSTMCell

    iterations_per_exp = 20
    # iterations_per_exp = 20

    time_steps = 250
    # time_steps = 100
    n_train = int(9e5)
    n_test = int(1e4)
    n_units = 90
    learning_rate = 1e-3
    decay = 0.9
    batch_size = 50
    GPU = 0
    memory = False
    adding = True
    activation = relu
    gate_activation = real_mod_sigmoid_beta
    subfolder = 'add_gate_study_t250_mod_sig_beta_real'
    gpu_mem_frac = 1.0
    qr_steps = -1
    stiefel = True
    real = True
    grad_clip = True

    # TODO remove!!
    # n_train = int(4e5)

    # Research hypothesis 1. Are gates helpling on the adding problem?
    # Run the gated case.
    experiments_gated = []
    for i in range(0, iterations_per_exp):
        cell_fun = cc.StiefelGatedRecurrentUnit
        #n_units = 80
        res = main(time_steps, n_train, n_test, n_units, learning_rate, decay,
                   batch_size, GPU, memory, adding, cell_fun, activation, gate_activation,
                   subfolder, gpu_mem_frac, qr_steps, stiefel, real, grad_clip)
        np_loss_train, np_loss_test, train_plot, test_losses = res
        print('gated experiment', i, 'done')
        experiments_gated.append([np_loss_train, np_loss_test])
    experiments_gated = np.array(experiments_gated)
    # Run the ungated case.
    if 0:
        experiments_no_gates = []
        for i in range(0, iterations_per_exp):
            n_units = 140
            cell_fun = cc.UnitaryCell
            res = main(time_steps, n_train, n_test, n_units, learning_rate, decay,
                       batch_size, GPU, memory, adding, cell_fun, activation,
                       gate_activation, subfolder, gpu_mem_frac, qr_steps, stiefel,
                       real, grad_clip)
            np_loss_train, np_loss_test, train_plot, test_losses = res
            print('ungated experiment', i, 'done')
            experiments_no_gates.append([np_loss_train, np_loss_test])
        experiments_no_gates = np.array(experiments_no_gates)

        # test on the last iterations:
        t0, p0 = scistats.ttest_ind(a=experiments_gated[:, 0],
                                    b=experiments_no_gates[:, 0])
        print('train mean gated:', np.mean(experiments_gated[:, 0]),
              'train mean no gates:', np.mean(experiments_no_gates[:, 0]))
        print('t and p on training', t0, p0)

        print('test mean gated:', np.mean(experiments_gated[:, 1]),
              'test mean no gates:', np.mean(experiments_no_gates[:, 1]))
        t1, p1 = scistats.ttest_ind(a=experiments_gated[:, 1],
                                    b=experiments_no_gates[:, 1])
        print('t and p on test', t1, p1)

        to_dump = [experiments_gated, experiments_no_gates, t0, p0, t1, p1]
        pickle.dump(to_dump, open('logs/' + subfolder + '/test_res.pkl', "wb"))
    else:
        to_dump = [experiments_gated]
        pickle.dump(to_dump, open('logs/' + subfolder + '/test_res.pkl', "wb"))
