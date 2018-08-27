print(' Test print.')
import sys
import time
import pickle
import numpy as np
import tensorflow as tf
from sklearn.metrics import average_precision_score

from custom_conv import complex_conv1D
from music_net_handler import MusicNet
from IPython.core.debugger import Tracer
debug_here = Tracer()
print('Does this get printed')
sys.path.insert(0, "../")
import custom_cells as cc

print('CNN experiment started.')
subfolder = 'CNN'

m = 128         # number of notes
fs = 11000      # samples/second
features_idx = 0    # first element of (X,Y) data tuple
labels_idx = 1      # second element of (X,Y) data tuple

# Network parameters:
c = 1            # number of context vectors
batch_size = 10      # The number of data points to be processed in parallel.
d = 256              # CNN filter depth.
filter_width = 256   # CNN filter length
stride = 16
filter_width_2 = 64
stride2 = 8

# FFT parameters:
# window_size = 16384
window_size = 8192
# window_size = 2048


# Training parameters:
learning_rate = 0.001
learning_rate_decay = 0.9
decay_iterations = 10000
iterations = 250000
# iterations = 10000
GPU = [6]


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


print('Setting up the tensorflow graph.')
train_graph = tf.Graph()
with train_graph.as_default():
    global_step = tf.Variable(0, trainable=False, name='global_step')
    # We use c input windows to give the RNN acces to context.
    x = tf.placeholder(tf.float32, shape=[batch_size, window_size])
    # The ground labelling is used during traning, wich random sampling
    # from the network output.
    y_gt = tf.placeholder(tf.float32, shape=[batch_size, m])

    # compute the fft in the time domain data.
    # x = tf.spectral.fft(tf.complex(x, tf.zeros_like(x)))
    xf = tf.spectral.rfft(x)
    xf = tf.expand_dims(xf, -1)

    dec_learning_rate = tf.train.exponential_decay(learning_rate, global_step,
                                                   decay_iterations, learning_rate_decay,
                                                   staircase=True)
    optimizer = tf.train.RMSPropOptimizer(dec_learning_rate)
    tf.summary.scalar('learning_rate', dec_learning_rate)

    with tf.variable_scope('complex_CNN'):
        conv1 = complex_conv1D(xf, filter_width=filter_width, depth=d, stride=stride,
                               padding='VALID', scope='_layer1')
        # conv1 = cc.hirose(conv1, 'mod_relu_1')
        conv1 = cc.split_relu(conv1)
        # conv1 = tf.nn.relu(conv1)
        debug_here()
        conv2 = complex_conv1D(conv1, filter_width=filter_width_2, depth=d,
                               stride=stride2, padding='VALID', scope='_layer2')
        # conv2 = cc.hirose(conv2, 'mod_relu_2')
        # conv2 = tf.nn.relu(conv2)
        conv2 = cc.split_relu(conv2)
        debug_here()
        print('conv2-shape:', conv2)
        flat = tf.reshape(conv2, [batch_size, -1])
        y = tf.nn.sigmoid(cc.C_to_R(flat, m, reuse=None))
        # y = tf.nn.sigmoid(cc.matmul_plus_bias(flat, m, reuse=None, scope='fc'))
        L = tf.losses.mean_squared_error(y, y_gt)
    tf.summary.scalar('mean_squared_error', L)
    gvs = optimizer.compute_gradients(L)
    with tf.variable_scope("gradient_clipping"):
        capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
        # capped_gvs = [(tf.clip_by_norm(grad, 2.0), var) for grad, var in gvs]
        # loss = tf.Print(loss, [tf.reduce_mean(gvs[0]) for gv in gvs])
        training_step = optimizer.apply_gradients(capped_gvs,
                                                  global_step=global_step)
    init_op = tf.global_variables_initializer()
    summary_op = tf.summary.merge_all()
    saver = tf.train.Saver()
    test_summary = tf.summary.scalar('test_mse', L)
    parameter_total = compute_parameter_total(tf.trainable_variables())

# Load the data.
print('Loading music-Net...')
musicNet = MusicNet(c, window_size, window_size, sampling_rate=fs)
batched_time_music_lst, batcheded_time_labels_lst = musicNet.get_test_batches(batch_size)

print('parameters:', m, fs, features_idx, labels_idx, c, batch_size, filter_width,
      filter_width_2, d, window_size, stride, stride2, learning_rate, learning_rate_decay,
      iterations, GPU, parameter_total)

time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
param_str = 'lr_' + str(learning_rate) + '_lrd_' + str(learning_rate_decay) \
            + '_lrdi_' + str(decay_iterations) \
            + '_bs_' + str(batch_size) + '_ws_' + str(window_size) + '_fs_' + str(fs) \
            + '_depth_' + str(d)\
            + '_fw1_' + str(filter_width) + '_fw2_' + str(filter_width_2) \
            + '_str1_' + str(stride) + '_str2_' + str(stride2) \
            + '_layers_' + str(1) + '_loss_' + str(L.name[:-8]) \
            + '_totparam_' + str(parameter_total)
savedir = './logs' + '/' + subfolder + '/' + time_str \
          + '_' + param_str
summary_writer = tf.summary.FileWriter(savedir, graph=train_graph)

square_error = []
average_precision = []
gpu_options = tf.GPUOptions(visible_device_list=str(GPU)[1:-1])
config = tf.ConfigProto(allow_soft_placement=True,
                        log_device_placement=False,
                        gpu_options=gpu_options)
with tf.Session(graph=train_graph, config=config) as sess:
    start = time.time()
    print('Initialize...')
    init_op.run(session=sess)

    print('Training...')
    for i in range(iterations):
        if i % 100 == 0 and (i != 0 or len(square_error) == 0):
            batch_time_music_test, batched_time_labels_test = \
                musicNet.get_batch(musicNet.test_data, musicNet.test_ids,
                                   batch_size)
            feed_dict = {x: np.squeeze(batch_time_music_test, axis=1),
                         y_gt: np.squeeze(batched_time_labels_test, axis=1)}
            L_np, test_summary_eval, global_step_eval = sess.run([L, test_summary,
                                                                 global_step],
                                                                 feed_dict=feed_dict)
            square_error.append(L_np)
            summary_writer.add_summary(test_summary_eval, global_step=global_step_eval)

        if i % 5000 == 0:
            # run trough the entire test set.
            yflat = np.array([])
            yhatflat = np.array([])
            losses_lst = []
            for j in range(len(batched_time_music_lst)):
                batch_time_music = batched_time_music_lst[j]
                batched_time_labels = batcheded_time_labels_lst[j]
                feed_dict = {x: np.squeeze(batch_time_music, 1),
                             y_gt: np.squeeze(batched_time_labels, 1)}
                loss, Yhattest, np_global_step =  \
                    sess.run([L, y, global_step], feed_dict=feed_dict)
                yhatflat = np.append(yhatflat, Yhattest.flatten())
                yflat = np.append(yflat, batched_time_labels[:, -1, :].flatten())
                losses_lst.append(loss)
            average_precision.append(average_precision_score(yflat,
                                                             yhatflat))
            end = time.time()
            print(i, '\t', round(np.mean(losses_lst), 8),
                     '\t', round(average_precision[-1], 8),
                     '\t', round(end-start, 8))
            saver.save(sess, savedir + '/weights', global_step=np_global_step)
            start = time.time()

        batch_time_music, batched_time_labels = \
            musicNet.get_batch(musicNet.train_data, musicNet.train_ids, batch_size)
        feed_dict = {x: np.squeeze(batch_time_music, 1),
                     y_gt: np.squeeze(batched_time_labels, 1)}
        loss, out_net, out_gt, _, summaries, np_global_step = \
            sess.run([L, y, y_gt, training_step, summary_op, global_step],
                     feed_dict=feed_dict)
        summary_writer.add_summary(summaries, global_step=np_global_step)

    # save the network
    saver.save(sess, savedir + '/weights', global_step=np_global_step)
    pickle.dump(average_precision, open(savedir + "avgprec.pkl", "wb"))
