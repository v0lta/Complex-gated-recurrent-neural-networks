# Recreation of the Montreal adding problem experiments from Arjovski et al.
# Working with Tensorflow 1.3
import time
import numpy as np
import tensorflow as tf
import custom_cells as cc

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


time_steps = 100
n_train = int(2e6)
n_test = int(1e4)
n_units = 512
learning_rate = 1e-3
decay = 0.9
batch_size = 50
train_iterations = int(n_train/batch_size)
test_iterations = int(n_test/batch_size)
GPU = 0
memory = False
adding = True

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
    # cell = tf.contrib.rnn.LSTMCell(n_units, num_proj=output_size)
    # cell = cc.UnitaryCell(num_units=n_units, output_size=output_size)
    # cell = cc.UnitaryMemoryCell(num_units=n_units, output_size=output_size)

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

    optimizer = tf.train.RMSPropOptimizer(learning_rate, decay=decay)
    # with tf.variable_scope("gradient_clipping"):
    #     gvs = optimizer.compute_gradients(loss)
    #     # print(gvs)
    #     capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
    #     # loss = tf.Print(loss, [tf.reduce_mean(gvs[0]) for gv in gvs])
    #     train_op = optimizer.apply_gradients(capped_gvs)

    train_op = optimizer.minimize(loss)
    init_op = tf.global_variables_initializer()
    summary_op = tf.summary.merge_all()

gpu_options = tf.GPUOptions(visible_device_list=str(GPU))
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
summary_writer = tf.summary.FileWriter('logs/' + time_str + param_str, graph=graph)
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
