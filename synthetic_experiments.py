# pylint: disable=E722

# Recreation of the Montreal adding problem experiments from Arjovski et al.
# Working with Tensorflow 1.3
import os
import time
import numpy as np
import tensorflow as tf
import custom_cells as cc
import GRU_wrapper as wg

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


from custom_optimizers import RMSpropNatGrad

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
        # print('var_name', variable.name, 'shape', shape, 'dim', len(shape))
        variable_parameters = 1
        for dim in shape:
            # print(dim)
            variable_parameters *= dim.value
        # print('parameters', variable_parameters)
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


def main(time_steps, n_train, n_test, n_units, learning_rate, decay,
         batch_size, GPU, memory, adding,
         cell_fun, activation, gate_activation, subfolder, gpu_mem_frac,
         qr_steps, stiefel, real, grad_clip, single_gate=False):
    """
    This main function does all the experimentation.
    """
    print('params', time_steps, n_train, n_test, n_units, learning_rate, decay,
          batch_size, GPU, memory, adding,
          cell_fun, activation, subfolder, gpu_mem_frac,
          qr_steps, stiefel, real, grad_clip)
    train_iterations = int(n_train/batch_size)
    test_iterations = int(n_test/batch_size)
    print("Train iterations:", train_iterations)
    if memory:
        # following https://github.com/amarshah/complex_RNN/blob/master/memory_problem.py
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
                            activation=activation, real=real)
        elif cell_fun.__name__ == 'StiefelGatedRecurrentUnit':
            cell = cell_fun(num_units=n_units, num_proj=output_size,
                            activation=activation, gate_activation=gate_activation,
                            stiefel=stiefel,
                            real=real, single_gate=single_gate)
        elif cell_fun.__name__ == 'GRUCell':
            cell = wg.RealGRUWrapper(cell_fun(num_units=n_units), output_size)
        else:
            cell = cell_fun(num_units=n_units, num_proj=output_size,
                            use_peepholes=True)

        if adding:
            x = tf.placeholder(tf.float32, shape=(batch_size, time_steps, 2))
            y = tf.placeholder(tf.float32, shape=(batch_size, 1))
            y_hat = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)
            y_hat = y_hat[0]  # throw away the final state.
            y_hat = y_hat[:, -1, :]  # only the final output is interesting.
            loss = tf.losses.mean_squared_error(y, y_hat)
            tf.summary.scalar('mse', loss)

        if memory:
            x = tf.placeholder(tf.int32, shape=(batch_size, time_steps+20))
            y = tf.placeholder(tf.int32, shape=(batch_size, time_steps+20))
            x_hot = tf.one_hot(x, output_size, dtype=tf.float32)
            y_hat = tf.nn.dynamic_rnn(cell, x_hot, dtype=tf.float32)
            y_hat = y_hat[0]
            loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=y_hat, labels=y))
            tf.summary.scalar('cross_entropy', loss)
        optimizer = RMSpropNatGrad(learning_rate=learning_rate, decay=decay,
                                   global_step=global_step, qr_steps=qr_steps)
        if grad_clip:
            with tf.variable_scope("gradient_clipping"):
                gvs = optimizer.compute_gradients(loss)
                capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
                train_op = optimizer.apply_gradients(capped_gvs, global_step=global_step)
        else:
            train_op = optimizer.minimize(loss, global_step=global_step)
        init_op = tf.global_variables_initializer()
        summary_op = tf.summary.merge_all()
        parameter_total = compute_parameter_total(tf.trainable_variables())

    # choose the GPU to use and how much memory we require.
    gpu_options = tf.GPUOptions(visible_device_list=str(GPU),
                                per_process_gpu_memory_fraction=gpu_mem_frac)
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
        + '_' + str(batch_size) + '_clipping_' + str(grad_clip)
    if cell.__class__.__name__ is "UnitaryCell" or \
       cell.__class__.__name__ is "StiefelGatedRecurrentUnit":
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
            np_loss_train, summary_mem, np_global_step, _ =  \
                sess.run(run_lst, feed_dict=feed_dict)
            toc = time.time()
            if i % 25 == 0:
                print('iteration', i/100, '*10^2',
                      np.array2string(np.array(np_loss_train),
                                      precision=4),
                      'Baseline', np.array2string(np.array(baseline), precision=4),
                      'update took:', np.array2string(np.array(toc - tic), precision=4),
                      's')
            train_plot.append([i/100, np_loss_train])
            summary_writer.add_summary(summary_mem, global_step=np_global_step)

        test_losses = []
        for j in range(test_iterations):
            if memory:
                x_batch = test_data[0][(j)*batch_size:(j+1)*batch_size, :]
                y_batch = test_data[1][(j)*batch_size:(j+1)*batch_size, :]
                feed_dict = {x: x_batch,
                             y: y_batch}
            if adding:
                x_batch = test_data[0][:, (j)*batch_size:(j+1)*batch_size, :]
                y_batch = test_data[1][(j)*batch_size:(j+1)*batch_size, :]
                feed_dict = {x: np.transpose(x_batch, (1, 0, 2)),
                             y: y_batch}
            np_loss_test = sess.run([loss], feed_dict=feed_dict)
            test_losses.append(np_loss_test)
        print('test loss', np.mean(test_losses))
    summary_writer.close()
    return np_loss_train, np_loss_test[0], train_plot, test_losses
