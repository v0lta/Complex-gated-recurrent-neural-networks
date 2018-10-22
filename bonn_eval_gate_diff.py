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
from custom_cells import mod_sigmoid_gamma
from custom_cells import double_sigmoid
from custom_cells import single_sigmoid_real
from custom_cells import single_sigmoid_imag


from synthetic_experiments import main
from IPython.core.debugger import Tracer

debug_here = Tracer()

if __name__ == "__main__":
    # time_steps=100, n_train=int(2e6), n_test=int(1e4),
    # n_units=512, learning_rate=1e-3, decay=0.9,
    # batch_size=50, GPU=0, memory=False, adding=True,
    # cell_fun=tf.contrib.rnn.LSTMCell

    # iterations_per_exp = 2
    iterations_per_exp = 20

    time_steps = 250
    # time_steps = 100
    n_train = int(9e5)
    n_test = int(1e4)
    n_units = 80
    learning_rate = 1e-3
    decay = 0.9
    batch_size = 50
    GPU = 0
    memory = True
    adding = False
    activation = mod_relu
    cell_fun = cc.StiefelGatedRecurrentUnit
    subfolder = 'gate_variation_study_test_bk_2'
    gpu_mem_frac = 1.0
    qr_steps = -1
    stiefel = True
    real = False
    grad_clip = True

    # TODO remove!!
    # n_train = int(1e3)

    # Research hypothesis 2. Which complex gating function performs best?
    # Run the gated case.
    gate_act_lst = ['single_gate', gate_phase_hirose, mod_sigmoid_prod, mod_sigmoid_sum,
                    mod_sigmoid, mod_sigmoid_beta, mod_sigmoid_gamma]
    experiments = []
    for gate_act in gate_act_lst:
        experiments_gated = []
        for i in range(0, iterations_per_exp):
            if gate_act is 'single_gate':
                single_gate = True
                gate_activation = None
                res = main(time_steps, n_train, n_test, n_units, learning_rate, decay,
                           batch_size, GPU, memory, adding, cell_fun,
                           activation, gate_activation,
                           subfolder, gpu_mem_frac, qr_steps, stiefel, real, grad_clip,
                           single_gate)
                np_loss_train, np_loss_test, train_plot, test_losses = res
            else:
                single_gate = False
                gate_activation = gate_act
                res = main(time_steps, n_train, n_test, n_units, learning_rate, decay,
                           batch_size, GPU, memory, adding, cell_fun,
                           activation, gate_activation,
                           subfolder, gpu_mem_frac, qr_steps, stiefel, real, grad_clip,
                           single_gate)
                np_loss_train, np_loss_test, train_plot, test_losses = res
            print('experiment', i, 'done')
            experiments_gated.append([np_loss_train, np_loss_test])
        experiments.append(experiments_gated)

    to_dump = [experiments]
    pickle.dump(to_dump, open('logs/' + subfolder + '/exp_res.pkl', "wb"))

    # compute the means.
    means = []
    means_test = []
    for experiment in experiments:
        experiment = np.array(experiment)
        m_train = np.mean(experiment[:, 0])
        m_test = np.mean(experiment[:, 1])
        means.append(m_train)
        means_test.append(m_test)
