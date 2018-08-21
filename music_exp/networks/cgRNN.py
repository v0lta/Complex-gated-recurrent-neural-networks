# Do the imports.
import sys
import time
import numpy as np
import tensorflow as tf
# from scipy.fftpack import fft
from sklearn.metrics import average_precision_score
from IPython.core.debugger import Tracer
sys.path.insert(0, "../")
import custom_cells as cc
# import custom_optimizers as co
debug_here = Tracer()


# where to store the logfiles.
subfolder = 'test'

# Load the data.
d = 2048        # input dimensions -> Window size
m = 128         # number of notes
fs = 44100      # samples/second
features_idx = 0    # first element of (X,Y) data tuple
labels_idx = 1      # second element of (X,Y) data tuple

# RNN parameters:
c = 24            # number of context vectors
cell_size = 2048  # complex RNN cell state size
stiefel = False   # Do not use stiefel weight normalization.
batch_size = 50   # The number of data points to be processed in parallel.

# FFT parameters:
window_size = d
stride = 512

# Training parameters:
GPU = 7
learning_rate = 0.0001
learning_rate_decay = 0.9
iterations = 250000
concat_y = True
sample_prob_y = 0.8

# Warning: the full dataset is over 40GB. Make sure you have enough RAM!
# This can take a few minutes to load
train_data = dict(np.load(open('numpy/musicnet.npz', 'rb'), encoding='latin1'))

print('musicnet loaded.')
# split our the test set
test_data = dict()
for id in (2303, 2382, 1819):  # test set
    test_data[str(id)] = train_data.pop(str(id))

train_ids = list(train_data.keys())
test_ids = list(test_data.keys())

print('splitting done.')
print(len(train_data))
print(len(test_data))


# data selection funciton
def select(data, index, window, c):
    time_music = []
    labels = []
    for cx in range(0, c):
            start = index - (c - cx)*stride
            center = start + int(window_size/2)
            end = start + window_size
            time_data = data[features_idx][start:end]
            # label stuff that's on in the center of the window
            label_vec = np.zeros([m])
            for active_label in data[labels_idx][center]:
                label_vec[active_label.data[1]] = 1
            time_music.append(time_data)
            labels.append(label_vec)
    return np.array(time_music), np.array(labels)


def get_batch(data, data_indices, batch_size):
    """
    Get a training batch.
    Args:
        data: Dictionary {file_id, time_domain_numpy_array}
        data_indices: The file_id dictionary keys for data.
        batch_size: The batch size used in the graph.
    Returns:
        batch_time_music: (batch_size, c, d) array with time
                          domain data.
        batched_time_labels: (batch_size, c, m) array labels.
    """
    batch_time_music = []
    batched_time_labels = []
    batched = 0
    while batched < batch_size:
        # select a random recording from the data-set.
        dat_idx = np.random.randint(0, len(data_indices))
        # go to a random position in the recording.
        record_with_label = data[data_indices[dat_idx]]
        offset = d/2 + c*window_size
        rec_idx = np.random.randint(offset, len(record_with_label[features_idx])-d/2)
        time_music, labels = select(record_with_label, rec_idx, window_size, c)
        if time_music.shape == (c, d) and labels.shape == (c, m):
            batch_time_music.append(time_music)
            batched_time_labels.append(labels)
            batched += 1
        else:
            pass
            # print('skipping sample.')

    batch_time_music = np.array(batch_time_music)
    batched_time_labels = np.array(batched_time_labels)

    # check the shapes.
    assert (batch_time_music.shape == (batch_size, c, d)
            and batched_time_labels.shape == (batch_size, c, m))
    return batch_time_music, batched_time_labels


def get_test_batches(data, data_indices, batch_size):
    """
    Set up the test set lists.
    """
    Xtest = []
    Ytest = []
    for dat_idx in data_indices:
        for j in range(7500):
            # start from one second to give us some room for larger segments
            rec_idx = fs+j*512
            record_with_label = data[dat_idx]
            time_music, labels = select(record_with_label, rec_idx, window_size, c)
            Xtest.append(time_music)
            Ytest.append(labels)
    Xtest = np.array(Xtest)
    Ytest = np.array(Ytest)
    # Reshape and check the shapes.
    batched_time_music_lst = np.split(Xtest, int(Xtest.shape[0]/batch_size), axis=0)
    batcheded_time_labels_lst = np.split(Ytest, int(Xtest.shape[0]/batch_size), axis=0)
    assert len(batched_time_music_lst) == len(batcheded_time_labels_lst)
    return batched_time_music_lst, batcheded_time_labels_lst


batched_time_music_lst, batcheded_time_labels_lst = get_test_batches(test_data, test_ids,
                                                                     batch_size)


print('setting up the tensorflow graph.')
train_graph = tf.Graph()
with train_graph.as_default():
    global_step = tf.Variable(0, trainable=False, name='global_step')
    # We use c input windows to give the RNN acces to context.
    x = tf.placeholder(tf.float32, shape=[batch_size, c, d])
    # The ground labelling is used during traning, wich random sampling
    # from the network output.
    y_ = tf.placeholder(tf.float32, shape=[batch_size, c, m])

    # compute the fft in the time domain data.
    # x = tf.spectral.fft(tf.complex(x, tf.zeros_like(x)))
    xf = tf.spectral.rfft(x)

    # TODO: concatenate y_ and write an,
    # RNN wrapper to sample from the RNN output.

    # initial_state = tf.get_variable('initial_state', [batch_size, cell_size])
    cell = cc.StiefelGatedRecurrentUnit(num_units=cell_size, stiefel=stiefel,
                                        num_proj=m, complex_input=True)
    y, final_state = tf.nn.dynamic_rnn(cell, xf, dtype=tf.float32)
    # L = tf.losses.sigmoid_cross_entropy(y[:, -1, :], y_[:, -1, :])
    # L = tf.reduce_mean(tf.nn.l2_loss(y[:, -1, :] - y_[:, -1, :]))
    L = tf.losses.mean_squared_error(y_[:, -1, :], y[:, -1, :])
    tf.summary.scalar('mean_squared_error', L)

    dec_learning_rate = tf.train.exponential_decay(learning_rate, global_step,
                                                   50000, learning_rate_decay,
                                                   staircase=True)
    optimizer = tf.train.RMSPropOptimizer(dec_learning_rate)
    tf.summary.scalar('learning_rate', dec_learning_rate)
    gvs = optimizer.compute_gradients(L)
    # print(gvs)
    capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
    # capped_gvs = [(tf.clip_by_norm(grad, 2.0), var) for grad, var in gvs]
    # loss = tf.Print(loss, [tf.reduce_mean(gvs[0]) for gv in gvs])
    training_step = optimizer.apply_gradients(capped_gvs,
                                              global_step=global_step)
    # training_step = optimizer.minimize(L)
    init_op = tf.global_variables_initializer()
    summary_op = tf.summary.merge_all()
    saver = tf.train.Saver()
    test_summary = tf.summary.scalar('test_mse', L)


time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
param_str = 'lr_' + str(learning_rate) + '_lrd_' + str(learning_rate_decay) \
            + '_size_' + str(cell_size) \
            + '_layers_' + str(1) + '_loss_' + str(L.name[:-8])
savedir = '../logs' + '/' + subfolder + '/' + time_str \
          + '_' + param_str
summary_writer = tf.summary.FileWriter(savedir, graph=train_graph)

square_error = []
average_precision = []
gpu_options = tf.GPUOptions(visible_device_list=str(GPU))
config = tf.ConfigProto(allow_soft_placement=True,
                        log_device_placement=False,
                        gpu_options=gpu_options)
with tf.Session(graph=train_graph, config=config) as sess:
    start = time.time()
    print('initialize')
    init_op.run(session=sess)

    print('Training...')
    for i in range(iterations):
        if i % 1000 == 0 and (i != 0 or len(square_error) == 0):
            batch_time_music_test, batched_time_labels_test = \
                get_batch(test_data, test_ids, batch_size)
            feed_dict = {x: batch_time_music_test, y_: batched_time_labels_test}
            L, test_summary_eval, global_step_eval = sess.run([L, test_summary,
                                                               global_step],
                                                              feed_dict=feed_dict)
            square_error.append(L)
            summary_writer.add_summary(test_summary_eval, global_step=global_step_eval)

        if i % 5000 == 0:
            # run trough the entire test set.
            yflat = np.array([])
            yhatflat = np.array([])
            losses_lst = []
            for i in range(len(batched_time_music_lst)):
                batch_time_music = batched_time_music_lst[i]
                batched_time_labels = batcheded_time_labels_lst[i]
                loss, Yhattest = sess.run([L, y], feed_dict={x: batch_time_music_test})
                yhatflat = np.append(yhatflat, Yhattest[:, -1, :].flatten())
                yflat = np.append(yflat, batched_time_labels[:, -1, :].flatten())
                losses_lst.append(loss)
            average_precision.append(average_precision_score(yflat,
                                                             yhatflat))
            end = time.time()
            print(i, '\t', round(np.mean(losses_lst), 8),
                     '\t', round(average_precision[-1], 8),
                     '\t', round(end-start, 8))
            start = time.time()
        batch_time_music, batched_time_labels = \
            get_batch(train_data, train_ids, batch_size)
        # debug_here()
        loss, out_net, out_gt, _, summaries, np_global_step = \
            sess.run([L, y, y_, training_step, summary_op, global_step],
                     feed_dict={x: batch_time_music, y_: batched_time_labels})
        summary_writer.add_summary(summaries, global_step=np_global_step)

    # save the network
    saver.save(sess, savedir)
