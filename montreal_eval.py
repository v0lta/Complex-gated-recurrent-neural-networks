# Recreation of the Montreal adding and memory problem experiments from Arjovski et al.
import tensorflow as tf
import argparse
import custom_cells as cc

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
# from custom_cells import real_mod_sigmoid_beta

from synthetic_experiments import main

if __name__ == "__main__":
    # time_steps=100, n_train=int(2e6), n_test=int(1e4),
    # n_units=512, learning_rate=1e-3, decay=0.9,
    # batch_size=50, GPU=0, memory=False, adding=True,
    # cell_fun=tf.contrib.rnn.LSTMCell
    parser = argparse.ArgumentParser(
        description="Run the montreal implementation \
         of the hochreiter RNN evaluation metrics.")
    parser.add_argument("--model", default='uRNN',
                        help='Model name: LSTM, GRU, uRNN, sGRU')
    parser.add_argument('--time_steps', '-time_steps', type=int, default=250,
                        help='problem length in time')
    parser.add_argument('--n_train', '-n_train', type=int, default=int(1e6),
                        help='training iteration number')
    parser.add_argument('--n_test', '-n_test', type=int, default=int(1e4),
                        help='training iteration number')
    parser.add_argument('--n_units', '-n_units', type=int, default=128,
                        help='hidden layer size')
    parser.add_argument('--learning_rate', '-learning_rate', type=float, default=1e-3,
                        help='graident descent step size')
    parser.add_argument('--decay', '-decay', type=int, default=0.9,
                        help='learning rate decay')
    parser.add_argument('--batch_size', '-batch_size', type=int, default=50,
                        help='Number of batches to be processed in parallel.')
    parser.add_argument('--GPU', '-GPU', type=int, default=0,
                        help='the number of the desired GPU.')
    parser.add_argument('--gpu_mem_frac', '-gpu_mem_frac', type=float, default=1.0,
                        help='Specify a gpu_mem_frac to use.')
    parser.add_argument('--memory', '-memory', type=str, default=False,
                        help='If true the data will come from the memory problem.')
    parser.add_argument('--adding', '-adding', type=str, default=True,
                        help='If true the data will come from the adding problem.')
    parser.add_argument('--subfolder', '-subfolder', type=str, default='exp1',
                        help='Specify a subfolder to use.')
    parser.add_argument('--non_linearity', '-non_linearity', type=str, default='mod_relu',
                        help='Specify the unitary linearity. Options are linar, mod_relu \
                              hirose, moebius, or loop to automatically run all options.')
    parser.add_argument('--gate_non_linearity', '-gate_non_linearity', type=str,
                        default='mod_sigmoid',
                        help='Specify the gate non linearity. Options are linar, mod_sigmoid_prod \
                              mod_sigmoid_sum, gate_phase_hirose, mod_sigmoid, \
                              mod_sigmoid_beta.')
    parser.add_argument('--qr_steps', '-qr_steps', type=int, default=int(-1),
                        help='Specify how often numerical errors should be corrected and \
                              the state related matrices reorthogonalized, \
                              -1 means never.')
    parser.add_argument('--real', '-real', type=str, default=False,
                        help='Run the real version of models, \
                        which also support complex numbers.')
    parser.add_argument('--stiefel', '-stiefel', type=str, default='True',
                        help='Turn stiefel manifold optimization in the sGRU on or off.')
    parser.add_argument('--grad_clip', '-grad_clip', type=str, default='True',
                        help='Use gradient clipping.')

    args = parser.parse_args()
    dict = vars(args)
    act_loop = False
    prob_loop = False
    time_loop = False
    # find and replace string arguments.
    for key in dict:
        if dict[key] == "False":
            dict[key] = False
        elif dict[key] == "True":
            dict[key] = True
        elif dict[key] == "LSTM":
            dict[key] = tf.contrib.rnn.LSTMCell
        elif dict[key] == "GRU":
            dict[key] = tf.contrib.rnn.GRUCell
        elif dict[key] == "uRNN":
            dict[key] = cc.UnitaryCell
        elif dict[key] == "sGRU":
            dict[key] = cc.StiefelGatedRecurrentUnit
        elif dict[key] == "linear":
            dict[key] = linear
        elif dict[key] == "mod_relu":
            dict[key] = mod_relu
        elif dict[key] == "hirose":
            dict[key] = hirose
        elif dict[key] == "moebius":
            dict[key] = moebius
        elif dict[key] == "relu":
            dict[key] = relu
        elif dict[key] == "split_relu":
            dict[key] = split_relu
        elif dict[key] == "z_relu":
            dict[key] = z_relu
        elif dict[key] == "tanh":
            dict[key] = tanh
        elif dict[key] == "mod_sigmoid_prod":
            dict[key] = mod_sigmoid_prod
        elif dict[key] == "gate_phase_hirose":
            dict[key] = gate_phase_hirose
        elif dict[key] == "mod_sigmoid":
            dict[key] = mod_sigmoid
        elif dict[key] == "mod_sigmoid_beta":
            dict[key] = mod_sigmoid_beta
        elif dict[key] == "mod_sigmoid_sum":
            dict[key] = mod_sigmoid_sum
        elif dict[key] == 'loop':
            if key == 'non_linearity':
                act_loop = True
            if key == 'adding' or key == 'memory':
                prob_loop = True
        elif dict[key] == -1:
            if key == 'time_steps':
                time_loop = True

    if act_loop and prob_loop and time_loop:
        # for time_it in [100, 250, 500, 1000]:
        for time_it in [250]:
            for problem in ['adding', 'memory']:
                if problem == 'adding':
                    adding_bool = True
                    memory_bool = False
                if problem == 'memory':
                    adding_bool = False
                    memory_bool = True
                for act in [mod_relu]:
                    kwargs = {'cell_fun': dict['model'],
                              'time_steps': time_it,
                              'n_train': dict['n_train'],
                              'n_test': dict['n_test'],
                              'n_units': dict['n_units'],
                              'learning_rate': dict['learning_rate'],
                              'decay': dict['decay'],
                              'batch_size': dict['batch_size'],
                              'gpu_mem_frac': dict['gpu_mem_frac'],
                              'GPU': dict['GPU'],
                              'memory': memory_bool,
                              'adding': adding_bool,
                              'subfolder': dict['subfolder'],
                              'activation': act,
                              'gate_activation': dict['gate_non_linearity'],
                              'qr_steps': dict['qr_steps'],
                              'stiefel': dict['stiefel'],
                              'real': dict['real'],
                              'grad_clip': dict['grad_clip']}
                    # try:
                    main(**kwargs)
                    # except:
                    #     print(bcolors.WARNING + 'Experiment', act, problem, time_it,
                    #           'diverged' + bcolors.ENDC)
                    if dict['model'] == tf.contrib.rnn.LSTMCell:
                        break
                    if dict['model'] == tf.contrib.rnn.GRUCell:
                        break
    elif prob_loop and time_loop:
        for time_it in [250]:
            for problem in ['adding', 'memory']:
                if problem == 'adding':
                    adding_bool = True
                    memory_bool = False
                if problem == 'memory':
                    adding_bool = False
                    memory_bool = True
                kwargs = {'cell_fun': dict['model'],
                          'time_steps': time_it,
                          'n_train': dict['n_train'],
                          'n_test': dict['n_test'],
                          'n_units': dict['n_units'],
                          'learning_rate': dict['learning_rate'],
                          'decay': dict['decay'],
                          'batch_size': dict['batch_size'],
                          'gpu_mem_frac': dict['gpu_mem_frac'],
                          'GPU': dict['GPU'],
                          'memory': memory_bool,
                          'adding': adding_bool,
                          'subfolder': dict['subfolder'],
                          'activation': dict['non_linearity'],
                          'gate_activation': dict['gate_non_linearity'],
                          'qr_steps': dict['qr_steps'],
                          'stiefel': dict['stiefel'],
                          'real': dict['real'],
                          'grad_clip': dict['grad_clip']}
                main(**kwargs)
    else:
        kwargs = {'cell_fun': dict['model'],
                  'time_steps': dict['time_steps'],
                  'n_train': dict['n_train'],
                  'n_test': dict['n_test'],
                  'n_units': dict['n_units'],
                  'learning_rate': dict['learning_rate'],
                  'decay': dict['decay'],
                  'batch_size': dict['batch_size'],
                  'gpu_mem_frac': dict['gpu_mem_frac'],
                  'GPU': dict['GPU'],
                  'memory': dict['memory'],
                  'adding': dict['adding'],
                  'subfolder': dict['subfolder'],
                  'activation': dict['non_linearity'],
                  'gate_activation': dict['gate_non_linearity'],
                  'qr_steps': dict['qr_steps'],
                  'stiefel': dict['stiefel'],
                  'real': dict['real'],
                  'grad_clip': dict['grad_clip']}

        # TODO: run multiple times for different sequence lengths.
        _, _, _, _, = main(**kwargs)
