# Recreation of the Montreal adding problem experiments from Arjovski et al.
# Working with Tensorflow 1.3
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
         subfolder='exp1', gpu_mem_frac=1.0):
    """
    This main function does all the experimentation.
    """

    train_iterations = int(n_train/batch_size)
    test_iterations = int(n_test/batch_size)
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
    # set up the rnn graph.
    graph = tf.Graph()
    with graph.as_default():
        # #### Cell selection. ####
        if cell_fun.__name__ == 'LSTMCell':
            cell = cell_fun(num_units=n_units, num_proj=output_size)
        else:
            cell = cell_fun(num_units=n_units, num_proj=output_size,
                            activation=activation)

        if adding:
            x = tf.placeholder(tf.float32, shape=(batch_size, time_steps, 2))
            y = tf.placeholder(tf.float32, shape=(batch_size, 1))
            y_hat = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)
            y_hat = y_hat[0]  # throw away the final state.
            y_hat = y_hat[:, -1, :]  # only the final output is interesting.
            loss = tf.losses.mean_squared_error(y, y_hat)
            loss_summary_op = tf.summary.scalar('mse', loss)

        if memory:
            x = tf.placeholder(tf.float32, shape=(batch_size, time_steps+20, 1))
            y = tf.placeholder(tf.int32, shape=(batch_size, time_steps+20))
            y_hat = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)
            y_hat = y_hat[0]
            loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=y_hat, labels=y))
            loss_summary_op = tf.summary.scalar('cross_entropy', loss)

        # optimizer = tf.train.RMSPropOptimizer(learning_rate, decay=decay)
        optimizer = RMSpropNatGrad(learning_rate, decay=decay)
        # with tf.variable_scope("gradient_clipping"):
        #     gvs = optimizer.compute_gradients(loss)
        #     # print(gvs)
        #     capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
        #     # loss = tf.Print(loss, [tf.reduce_mean(gvs[0]) for gv in gvs])
        #     train_op = optimizer.apply_gradients(capped_gvs)
        # debug_here()
        train_op = optimizer.minimize(loss)
        init_op = tf.global_variables_initializer()
        summary_op = tf.summary.merge_all()

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
    param_str = '_' + problem + '_' + str(time_steps) + '_' + str(n_train) \
        + '_' + str(n_test) + '_' + str(n_units) + '_' + str(learning_rate) \
        + '_' + str(batch_size) + '_' + cell._activation.__name__ \
        + '_' + cell.__class__.__name__
    summary_writer = tf.summary.FileWriter('logs' + '/' + subfolder + '/' + time_str
                                           + param_str, graph=graph)
    print(param_str)

    # and run it!
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
            run_lst = [loss, summary_op, train_op]
            np_loss, summary_mem, _ = sess.run(run_lst, feed_dict=feed_dict)
            print('iteration', i/100, '*10^2', np_loss, 'Baseline', baseline)
            train_plot.append([i/100, np_loss])
            summary_writer.add_summary(summary_mem, global_step=i)

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
                        help='Model name: LSTM, UNN, GUNN')
    parser.add_argument('--time_steps', '-time_steps', type=int, default=100,
                        help='Copying Problem delay')
    parser.add_argument('--n_train', '-n_train', type=int, default=int(1e6),
                        help='training iteration number')
    parser.add_argument('--n_test', '-n_test', type=int, default=int(1e4),
                        help='training iteration number')
    parser.add_argument('--n_units', '-n_units', type=int, default=512,
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

    args = parser.parse_args()
    dict = vars(args)
    act_loop = False
    # find and replace string arguments.
    for i in dict:
        if (dict[i] == "False"):
            dict[i] = False
        elif dict[i] == "True":
            dict[i] = True
        elif dict[i] == "LSTM":
            dict[i] = tf.contrib.rnn.LSTMCell
        elif dict[i] == "UNN":
            dict[i] = cc.UnitaryCell
        elif dict[i] == "GUNN":
            dict[i] = cc.UnitaryMemoryCell
        elif dict[i] == "linear":
            dict[i] = linear
        elif dict[i] == "mod_relu":
            dict[i] = mod_relu
        elif dict[i] == "hirose":
            dict[i] = hirose
        elif dict[i] == "moebius":
            dict[i] = moebius
        elif dict[i] == 'loop':
            act_loop = True

    if act_loop:
        for act in [linear, mod_relu, hirose, moebius]:
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
                      'activation': act}
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
                  'activation': dict['non_linearity']}

        # TODO: run multiple times for different sequence lengths.
        main(**kwargs)
