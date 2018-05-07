# pylint: disable=E722

# Recreation of the Montreal adding problem experiments from Arjovski et al.
# Working with Tensorflow 1.3
import os
import time
import numpy as np
import tensorflow as tf
import argparse
import custom_cells as cc

from custom_cells import mod_relu
from custom_cells import hirose
from custom_cells import linear
from custom_cells import moebius

from custom_optimizers import RMSpropNatGrad


from IPython.core.debugger import Tracer
debug_here = Tracer()
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def compute_parameter_total(trainable_variables):
    total_parameters = 0
    for variable in trainable_variables:
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        print('var_name', variable.name, 'shape', shape, 'dim', len(shape))
        variable_parameters = 1
        for dim in shape:
            # print(dim)
            variable_parameters *= dim.value
        print('parameters', variable_parameters)
        total_parameters += variable_parameters
    print('total:', total_parameters)
    return total_parameters


def generate_data_adding(time_steps, n_data):
    """
    Generate data for the adding problem.
    Source: https://github.com/amarshah/complex_RNN/blob/master/adding_problem.py
    Params:
        time_steps: The number of time steps we would like to consider.
        n_data: the number of sequences we would like to consider.
    returns:
        x: [time_steps, n_data, 2] input array.
        y: [n_data, 1] output array.
    """
    x = np.asarray(np.zeros((time_steps, int(n_data), 2)),
                   dtype=np.float)
    # this should be low=-1!? According to hochreiter et al?!
    x[:, :, 0] = np.asarray(np.random.uniform(low=0.,
                                              high=1.,
                                              size=(time_steps, n_data)),
                            dtype=np.float)
    inds = np.asarray(np.random.randint(time_steps/2, size=(n_data, 2)))
    inds[:, 1] += int(time_steps/2)

    for i in range(int(n_data)):
        x[inds[i, 0], i, 1] = 1.0
        x[inds[i, 1], i, 1] = 1.0

    y = (x[:, :, 0] * x[:, :, 1]).sum(axis=0)
    y = np.reshape(y, (n_data, 1))
    return x, y


def generate_data_memory(time_steps, n_data, n_sequence):
    seq = np.random.randint(1, high=9, size=(n_data, n_sequence))
    zeros1 = np.zeros((n_data, time_steps - 1))
    zeros2 = np.zeros((n_data, time_steps))
    marker = 9 * np.ones((n_data, 1))
    zeros3 = np.zeros((n_data, n_sequence))

    x = np.concatenate((seq, zeros1, marker, zeros3), axis=1).astype('int32')
    y = np.concatenate((zeros3, zeros2, seq), axis=1).astype('int32')
    return x, y


def main(time_steps=100, n_train=int(2e6), n_test=int(1e4),
         n_units=512, learning_rate=1e-3, decay=0.9,
         batch_size=50, GPU=0, memory=False, adding=True,
         cell_fun=tf.contrib.rnn.LSTMCell, activation=mod_relu,
         subfolder='exp1', gpu_mem_frac=1.0,
         qr_steps=-1, orthogonal=False, unitary=False):
    """
    This main function does all the experimentation.
    """

    train_iterations = int(n_train/batch_size)
    test_iterations = int(n_test/batch_size)
    print("Train iterations:", train_iterations)
    if memory:
        output_size = 9
        n_sequence = 10
        train_data = generate_data_memory(time_steps, n_train, n_sequence)
        test_data = generate_data_memory(time_steps, n_test, n_sequence)
        # --- baseline ----------------------
        baseline = np.log(8) * 10/(time_steps + 20)
        print("Baseline is " + str(baseline))
    elif adding:
        output_size = 1
        train_data = generate_data_adding(time_steps, n_train)
        test_data = generate_data_adding(time_steps, n_test)
        baseline = 0.167
    else:
        raise NotImplementedError()

    # ------------------------- set up the rnn graph. ---------------------------
    graph = tf.Graph()
    with graph.as_default():
        global_step = tf.Variable(0, trainable=False, name='global_step')
        # #### Cell selection. ####
        if cell_fun.__name__ == 'UnitaryCell':
            cell = cell_fun(num_units=n_units, num_proj=output_size,
                            activation=activation)
        elif cell_fun.__name__ == 'UnitaryMemoryCell':
            cell = cell_fun(num_units=n_units, num_proj=output_size,
                            activation=activation, orthogonal_gate=orthogonal,
                            unitary_gate=unitary)
        elif cell_fun.__name__ == 'ComplexGatedRecurrentUnit':
            cell = cell_fun(num_units=n_units, num_proj=output_size,
                            activation=activation)
        else:
            cell = cell_fun(num_units=n_units, num_proj=output_size)

        if adding:
            x = tf.placeholder(tf.float32, shape=(batch_size, time_steps, 2))
            y = tf.placeholder(tf.float32, shape=(batch_size, 1))
            y_hat = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)
            y_hat = y_hat[0]  # throw away the final state.
            y_hat = y_hat[:, -1, :]  # only the final output is interesting.
            loss = tf.losses.mean_squared_error(y, y_hat)
            tf.summary.scalar('mse', loss)

        if memory:
            x = tf.placeholder(tf.float32, shape=(batch_size, time_steps+20, 1))
            y = tf.placeholder(tf.int32, shape=(batch_size, time_steps+20))
            y_hat = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)
            y_hat = y_hat[0]
            loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=y_hat, labels=y))
            tf.summary.scalar('cross_entropy', loss)

        # optimizer = tf.train.RMSPropOptimizer(learning_rate, decay=decay)
        optimizer = RMSpropNatGrad(learning_rate=learning_rate, decay=decay,
                                   global_step=global_step, qr_steps=qr_steps)
        with tf.variable_scope("gradient_clipping"):
            gvs = optimizer.compute_gradients(loss)
            # print(gvs)
            capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
            # loss = tf.Print(loss, [tf.reduce_mean(gvs[0]) for gv in gvs])
            train_op = optimizer.apply_gradients(capped_gvs)
        # debug_here()
        train_op = optimizer.minimize(loss, global_step=global_step)
        init_op = tf.global_variables_initializer()
        summary_op = tf.summary.merge_all()
        parameter_total = compute_parameter_total(tf.trainable_variables())

    gpu_options = tf.GPUOptions(visible_device_list=str(GPU),
                                per_process_gpu_memory_fraction=gpu_mem_frac)
    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
    config = tf.ConfigProto(allow_soft_placement=True,
                            log_device_placement=False,
                            gpu_options=gpu_options)
    time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    if memory:
        problem = 'memory'
    if adding:
        problem = 'adding'
    param_str = problem + '_' + str(time_steps) + '_' + str(n_train) \
        + '_' + str(n_test) + '_' + str(n_units) + '_' + str(learning_rate) \
        + '_' + str(batch_size)
    # TODO. add statement checking if the nat grad optimizer is there.
    if cell.__class__.__name__ is "UnitaryCell" or \
       cell.__class__.__name__ is "UnitaryMemoryCell" or \
       cell.__class__.__name__ is "ComplexGatedRecurrentUnit":
        param_str += '_' + cell.to_string()
        param_str += '_' + 'nat_grad_rms' + '_' + str(optimizer._nat_grad_normalization)
        param_str += '_' + 'qr_steps' + '_' + str(optimizer._qr_steps)
    else:
        param_str += '_' + str(cell.__class__.__name__)

    # add parameter_total:
    param_str += '_' + 'pt' + '_' + str(parameter_total)

    summary_writer = tf.summary.FileWriter('logs' + '/' + subfolder + '/' + time_str
                                           + '_' + param_str, graph=graph)
    print(bcolors.OKGREEN + param_str + bcolors.ENDC)

    # ------------------------- and run it! ---------------------------------
    train_plot = []
    with tf.Session(graph=graph, config=config) as sess:
        init_op.run()
        for i in range(train_iterations):
            if memory:
                x_batch = train_data[0][(i)*batch_size:(i+1)*batch_size, :]
                x_batch = np.expand_dims(x_batch, -1)
                y_batch = train_data[1][(i)*batch_size:(i+1)*batch_size, :]
                feed_dict = {x: x_batch,
                             y: y_batch}
            if adding:
                x_batch = train_data[0][:, (i)*batch_size:(i+1)*batch_size, :]
                y_batch = train_data[1][(i)*batch_size:(i+1)*batch_size, :]
                feed_dict = {x: np.transpose(x_batch, (1, 0, 2)),
                             y: y_batch}
            run_lst = [loss, summary_op, global_step, train_op]
            tic = time.time()
            np_loss, summary_mem, np_global_step, _ =  \
                sess.run(run_lst, feed_dict=feed_dict)
            toc = time.time()
            print('iteration', i/100, '*10^2', np.array2string(np.array(np_loss),
                                                               precision=4),
                  'Baseline', np.array2string(np.array(baseline), precision=4),
                  'update took:', np.array2string(np.array(toc - tic), precision=4), 's')
            train_plot.append([i/100, np_loss])
            summary_writer.add_summary(summary_mem, global_step=np_global_step)

        test_losses = []
        for j in range(test_iterations):
            if memory:
                x_batch = test_data[0][(j)*batch_size:(j+1)*batch_size, :]
                x_batch = np.expand_dims(x_batch, -1)
                y_batch = test_data[1][(j)*batch_size:(j+1)*batch_size, :]
                feed_dict = {x: x_batch,
                             y: y_batch}
            if adding:
                x_batch = test_data[0][:, (j)*batch_size:(j+1)*batch_size, :]
                y_batch = test_data[1][(j)*batch_size:(j+1)*batch_size, :]
                feed_dict = {x: np.transpose(x_batch, (1, 0, 2)),
                             y: y_batch}
            np_loss = sess.run([loss], feed_dict=feed_dict)
            test_losses.append(np_loss)
        print('test loss', np.mean(test_losses))
    summary_writer.close()


if __name__ == "__main__":
    # time_steps=100, n_train=int(2e6), n_test=int(1e4),
    # n_units=512, learning_rate=1e-3, decay=0.9,
    # batch_size=50, GPU=0, memory=False, adding=True,
    # cell_fun=tf.contrib.rnn.LSTMCell
    parser = argparse.ArgumentParser(
        description="Run the montreal implementation \
         of the hochreiter RNN evaluation metrics.")
    parser.add_argument("--model", default='EUNN',
                        help='Model name: LSTM, UNN, GUNN, CGRU')
    parser.add_argument('--time_steps', '-time_steps', type=int, default=100,
                        help='Copying Problem delay')
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
    parser.add_argument('--non_linearity', '-non_linearity', type=str, default='linear',
                        help='Specify the unitary linearity. Options are linar, mod_relu \
                              hirose, moebius, or loop to automatically run all options.')
    parser.add_argument('--qr_steps', '-qr_steps', type=int, default=int(-1),
                        help='Specify how often numerical errors should be corrected and \
                              the state related matrices reorthogonalized, \
                              -1 means no qr.')
    parser.add_argument('--orthogonal', '-orthogonal', type=str, default='False',
                        help='Sould the memory cell gates be orthogonal.')
    parser.add_argument('--unitary', '-unitary', type=str, default='False',
                        help='Sould the memory cell gates be orthogonal.')

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
        elif dict[key] == "UNN":
            dict[key] = cc.UnitaryCell
        elif dict[key] == "GUNN":
            dict[key] = cc.UnitaryMemoryCell
        elif dict[key] == "CGRU":
            dict[key] = cc.ComplexGatedRecurrentUnit
        elif dict[key] == "linear":
            dict[key] = linear
        elif dict[key] == "mod_relu":
            dict[key] = mod_relu
        elif dict[key] == "hirose":
            dict[key] = hirose
        elif dict[key] == "moebius":
            dict[key] = moebius
        elif dict[key] == 'loop':
            if key == 'non_linearity':
                act_loop = True
            if key == 'adding' or key == 'memory':
                prob_loop = True
        elif dict[key] == -1:
            if key == 'time_steps':
                time_loop = True

    if act_loop and prob_loop and time_loop:
        for time_it in [100, 250, 500, 1000]:
            for problem in ['adding', 'memory']:
                if problem == 'adding':
                    adding_bool = True
                    memory_bool = False
                if problem == 'memory':
                    adding_bool = False
                    memory_bool = True
                for act in [linear, mod_relu, hirose, moebius]:
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
                              'qr_steps': dict['qr_steps'],
                              'orthogonal': dict['orthogonal'],
                              'unitary': dict['unitary']}
                    try:
                        main(**kwargs)
                    except:
                        print(bcolors.WARNING + 'Experiment', act, problem, time_it,
                              'diverged' + bcolors.ENDC)
                    if dict['model'] == tf.contrib.rnn.LSTMCell:
                        break

    elif act_loop and prob_loop:
        for act in [linear, mod_relu, hirose, moebius]:
            for problem in ['adding', 'memory']:
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
                          'memory': memory_bool,
                          'adding': adding_bool,
                          'subfolder': dict['subfolder'],
                          'activation': act,
                          'qr_steps': dict['qr_steps'],
                          'orthogonal': dict['orthogonal'],
                          'unitary': dict['unitary']}
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
                  'qr_steps': dict['qr_steps'],
                  'orthogonal': dict['orthogonal'],
                  'unitary': dict['unitary']}

        # TODO: run multiple times for different sequence lengths.
        main(**kwargs)
