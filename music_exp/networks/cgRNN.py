# Do the imports.
import sys
import time
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.core.framework import summary_pb2
import tensorflow.contrib.signal as tfsignal
from custom_conv import complex_conv1D
# from scipy.fftpack import fft
from sklearn.metrics import average_precision_score
from IPython.core.debugger import Tracer
from music_net_handler import MusicNet
sys.path.insert(0, "../")
import custom_cells as cc

# import custom_optimizers as co
debug_here = Tracer()


# where to store the logfiles.
subfolder = 'cgRNN_only'

m = 128         # number of notes
sampling_rate = 11000      # samples/second
features_idx = 0    # first element of (X,Y) data tuple
labels_idx = 1      # second element of (X,Y) data tuple

# Network parameters:
c = 24              # number of context vectors
batch_size = 5      # The number of data points to be processed in parallel.
# d = [64]          # CNN filter depth.
filter_width = [12, 9, 9, 6]  # cnn filter length
stride = [4, 3, 3, 2]
d = [16, 32, 64, 128]
# filter_width = [256]
# stride = [8]
# d = [32]
# d = [16, 32, 64, 64, 128, 128]      # CNN filter depth.
# filter_width = [6, 3, 3, 3, 3, 3]   # CNN filter length
# stride = [2, 2, 2, 2, 2, 1]
assert len(d) == len(filter_width)
assert len(filter_width) == len(stride)

dense_size = 1024   # dense layer shape.
cell_size = 1024    # cell depth.
CNN = False
RNN = True
stiefel = False
dropout = False

# FFT parameters:
# window_size = 16384
# window_size = 4096
window_size = 2048
fft_stride = 512

# Training parameters:
learning_rate = 0.0001
learning_rate_decay = 0.9
decay_iterations = 50000
iterations = 400000
GPU = [7]


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
    x = tf.placeholder(tf.float32, shape=[batch_size, c, window_size])
    # The ground labelling is used during traning, wich random sampling
    # from the network output.
    y_gt = tf.placeholder(tf.float32, shape=[batch_size, c, m])

    # compute the fft in the time domain data.
    # xf = tf.spectral.fft(tf.complex(x, tf.zeros_like(x)))
    w = tfsignal.hann_window(window_size, periodic=True)
    xf = tf.spectral.rfft(x*w)
    dec_learning_rate = tf.train.exponential_decay(learning_rate, global_step,
                                                   decay_iterations, learning_rate_decay,
                                                   staircase=True)
    optimizer = tf.train.RMSPropOptimizer(dec_learning_rate)
    tf.summary.scalar('learning_rate', dec_learning_rate)

    if CNN:
        with tf.variable_scope('complex_CNN'):
            xfd = tf.reshape(xf, [batch_size*c, -1])
            xfd = tf.expand_dims(xfd, -1)

            conv = [xfd]
            for layer_no, layer_d in enumerate(d):
                conv_tmp = complex_conv1D(conv[-1], filter_width=filter_width[layer_no],
                                          depth=layer_d, stride=stride[layer_no],
                                          padding='VALID', scope='_layer' + str(layer_no))
                conv.append(cc.split_relu(conv_tmp))
                print('conv2 shape', conv[-1].shape)
            flat = tf.reshape(conv[-1], [batch_size, c, -1])
            RNN_in = flat
    else:
        RNN_in = xf
    if RNN:
        def define_bidirecitonal(RNN_in, cell_size, dense_size,
                                 stiefel, dropout, reuse=None):
            cell = cc.StiefelGatedRecurrentUnit(num_units=cell_size, stiefel=stiefel,
                                                num_proj=None, complex_input=True,
                                                dropout=dropout, reuse=reuse)
            # Bidirectional RNN encoder.
            outputs, states = tf.nn.bidirectional_dynamic_rnn(
                cell, cell, RNN_in, dtype=tf.float32)

            # RNN decoder.
            if not dense_size:
                to_decode = tf.concat([tf.complex(outputs[0][:, :, :cell_size],
                                                  outputs[0][:, :, cell_size:]),
                                       tf.complex(outputs[1][:, :, :cell_size],
                                                  outputs[1][:, :, cell_size:])],
                                      axis=-1)
                # RNN decoder.
                decoder_cell = cc.StiefelGatedRecurrentUnit(
                    num_units=int(cell_size), stiefel=stiefel, num_proj=m,
                    complex_input=True, reuse=reuse)
                y, _ = tf.nn.dynamic_rnn(decoder_cell, to_decode,
                                         dtype=tf.float32)
            else:
                # dense matmul decoder.
                outputs = tf.concat(outputs, axis=-1)
                to_dense_shape = outputs.get_shape().as_list()
                to_dense = tf.reshape(outputs, [to_dense_shape[0]*to_dense_shape[1],
                                                to_dense_shape[2]])
                dense_out = tf.nn.relu(cc.matmul_plus_bias(to_dense, dense_size,
                                       'dense_layer', bias=True, reuse=reuse))
                y = cc.C_to_R(dense_out, m, reuse=reuse)
                y = tf.reshape(y, [to_dense_shape[0], to_dense_shape[1], -1])
            return y
        y = define_bidirecitonal(RNN_in, cell_size, dense_size,
                                 stiefel, dropout)
        if dropout:
            print('test part of graph.')
            y_test = define_bidirecitonal(RNN_in, cell_size, dense_size,
                                          stiefel, dropout=False, reuse=True)
        else:
            y_test = y
    else:
        if c != 1:
            raise ValueError("c must be one for non RNN networks.")

        full = cc.split_relu(cc.complex_matmul(y, dense_size, 'complex_dense',
                                               reuse=None, bias=True))
        y = tf.nn.sigmoid(cc.C_to_R(full, m, reuse=None))
        y = y[:, -1, :]
        y_gt = y_gt[:, -1, :]

    # L = tf.losses.sigmoid_cross_entropy(y[:, -1, :], y_[:, -1, :])
    # L = tf.reduce_mean(tf.nn.l2_loss(y[:, -1, :] - y_[:, -1, :]))
    # debug_here()
    L = tf.losses.sigmoid_cross_entropy(multi_class_labels=y_gt,
                                        logits=y)
    # L = tf.losses.mean_squared_error(y_gt, y)
    # L_test = tf.losses.mean_squared_error(y_gt, y_test)
    L_test = tf.losses.sigmoid_cross_entropy(multi_class_labels=y_gt,
                                             logits=y_test)
    gvs = optimizer.compute_gradients(L)
    tf.summary.scalar('train_l', L)
    # print(gvs)
    with tf.variable_scope("gradient_clipping"):
        capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
        # capped_gvs = [(tf.clip_by_norm(grad, 2.0), var) for grad, var in gvs]
        # loss = tf.Print(loss, [tf.reduce_mean(gvs[0]) for gv in gvs])
        training_step = optimizer.apply_gradients(capped_gvs,
                                                  global_step=global_step)
    # training_step = optimizer.minimize(L)
    init_op = tf.global_variables_initializer()
    summary_op = tf.summary.merge_all()
    saver = tf.train.Saver()
    test_summary = tf.summary.scalar('test_L', L_test)
    parameter_total = compute_parameter_total(tf.trainable_variables())

# Load the data.
# debug_here()
print('Loading music-Net...')
musicNet = MusicNet(c, fft_stride, window_size, sampling_rate=sampling_rate)
batched_time_music_lst, batcheded_time_labels_lst = musicNet.get_test_batches(batch_size)

print('parameters:', 'm', m, 'sampling_rate', sampling_rate, 'c', c,
      'batch_size', batch_size, 'filter_width', filter_width,
      'd', d, 'stride', stride, 'dense_size', dense_size,
      'window_size', window_size, 'fft_stride', fft_stride,
      'learning_rate', learning_rate,
      'learning_rate_decay', learning_rate_decay, 'iterations', iterations,
      'GPU', GPU, 'CNN', CNN, 'dropout', dropout, 'parameter_total', parameter_total)


def lst_to_str(lst):
    string = ''
    for lst_el in lst:
        string += str(lst_el) + '_'
    return string


time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
param_str = 'lr_' + str(learning_rate) + '_lrd_' + str(learning_rate_decay) \
            + '_lrdi_' + str(decay_iterations) + '_it_' + str(iterations) \
            + '_bs_' + str(batch_size) + '_ws_' + str(window_size) \
            + 'fft_stride' + str(fft_stride) + '_fs_' + str(sampling_rate)
if CNN:
    param_str += '_fw_' + lst_to_str(filter_width)  \
        + '_str_' + lst_to_str(stride) + '_depth_' + lst_to_str(d)
param_str += '_loss_' + str(L.name[:-8]) \
             + '_cnn_' + str(CNN) + '_dropout_' + str(dropout) \
             + '_cs_' + str(cell_size) + '_ds_' + str(dense_size) \
             + '_c_' + str(c) \
             + '_totparam_' + str(parameter_total)
savedir = './logs' + '/' + subfolder + '/' + time_str \
          + '_' + param_str
# debug_here()
print(savedir)
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
            feed_dict = {x: batch_time_music_test,
                         y_gt: batched_time_labels_test}
            L_np, test_summary_eval, global_step_eval = sess.run([L_test, test_summary,
                                                                 global_step],
                                                                 feed_dict=feed_dict)
            square_error.append(L_np)
            summary_writer.add_summary(test_summary_eval, global_step=global_step_eval)

        # if i % 5000 == 0:
        if i % 5000 == 0 and i > 0:
            # run trough the entire test set.
            yflat = np.array([])
            yhatflat = np.array([])
            losses_lst = []
            for j in range(len(batched_time_music_lst)):
                batch_time_music = batched_time_music_lst[j]
                batched_time_labels = batcheded_time_labels_lst[j]
                feed_dict = {x: batch_time_music,
                             y_gt: batched_time_labels}
                loss, Yhattest, np_global_step =  \
                    sess.run([L_test, y_test, global_step], feed_dict=feed_dict)
                losses_lst.append(loss)
                center = int(c/2.0)
                yhatflat = np.append(yhatflat, Yhattest[:, center, :].flatten())
                yflat = np.append(yflat, batched_time_labels[:, center, :].flatten())
            average_precision.append(average_precision_score(yflat,
                                                             yhatflat))
            end = time.time()
            print(i, '\t', round(np.mean(losses_lst), 8),
                     '\t', round(average_precision[-1], 8),
                     '\t', round(end-start, 8))
            saver.save(sess, savedir + '/weights', global_step=np_global_step)
            # add average precision to tensorboard...
            acc_value = summary_pb2.Summary.Value(tag="Accuracy",
                                                  simple_value=average_precision[-1])
            summary = summary_pb2.Summary(value=[acc_value])
            summary_writer.add_summary(summary, global_step=np_global_step)

            start = time.time()

        batch_time_music, batched_time_labels = \
            musicNet.get_batch(musicNet.train_data, musicNet.train_ids, batch_size)
        feed_dict = {x: batch_time_music,
                     y_gt: batched_time_labels}
        loss, out_net, out_gt, _, summaries, np_global_step = \
            sess.run([L, y, y_gt, training_step, summary_op, global_step],
                     feed_dict=feed_dict)
        summary_writer.add_summary(summaries, global_step=np_global_step)
        # if i % 10 == 0:
        #     print('loss', loss)

    # save the network
    saver.save(sess, savedir + '/weights/', global_step=np_global_step)
    pickle.dump(average_precision, open(savedir + "/avgprec.pkl", "wb"))
