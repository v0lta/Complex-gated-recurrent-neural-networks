# Do the imports.
import sys
from time import time
import numpy as np
import tensorflow as tf
# from scipy.fftpack import fft
from sklearn.metrics import average_precision_score
from IPython.core.debugger import Tracer
sys.path.insert(0, "../../")
import custom_cells as cc
import custom_optimizers as co
debug_here = Tracer()

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
batch_size = 48   # The number of data points to be processed in parallel.

# FFT parameters:
window_size = d
stride = 512

# Training parameters:
GPU = 3
learning_rate = 0.0001
iterations = 250000

# Warning: the full dataset is over 40GB. Make sure you have enough RAM!
# This can take a few minutes to load
train_data = dict(np.load(open('../numpy/musicnet.npz', 'rb'), encoding='latin1'))

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


# # create the test set we want 11 evenly spaced samples + context.
# test_set = []
# for test_id in test_ids:
#     for j in range(7500):
#         # start from one second to give us some room for larger segments
#         index = fs+j*stride
#         test_samples = select(test_data[test_id], index, window_size, c)
#         test_set.append(test_samples)


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
    L = tf.reduce_mean(tf.nn.l2_loss(y[:, -1, :] - y_[:, -1, :]))
    optimizer = tf.train.RMSPropOptimizer(learning_rate)
    gvs = optimizer.compute_gradients(L)
    # print(gvs)
    capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
    # capped_gvs = [(tf.clip_by_norm(grad, 2.0), var) for grad, var in gvs]
    # loss = tf.Print(loss, [tf.reduce_mean(gvs[0]) for gv in gvs])
    training_step = optimizer.apply_gradients(capped_gvs,
                                              global_step=global_step)
    # training_step = optimizer.minimize(L)
    init_op = tf.global_variables_initializer()


def get_batch(data, data_indices, batch_size):
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

    batch_time_music = np.array(batch_time_music)
    batched_time_labels = np.array(batched_time_labels)

    # check the shapes.
    assert (batch_time_music.shape == (batch_size, c, d)
            and batched_time_labels.shape == (batch_size, c, m))
    return batch_time_music, batched_time_labels


square_error = []
average_precision = []
gpu_options = tf.GPUOptions(visible_device_list=str(GPU))
config = tf.ConfigProto(allow_soft_placement=True,
                        log_device_placement=False,
                        gpu_options=gpu_options)
with tf.Session(graph=train_graph, config=config) as sess:
    start = time()
    print('initialize')
    init_op.run(session=sess)

    print('Training...')
    for i in range(iterations):
        if i % 1000 == 0 and (i != 0 or len(square_error) == 0):
            batch_time_music_test, batched_time_labels_test = \
                get_batch(test_data, test_ids, batch_size)
            square_error.append(sess.run(L, feed_dict={x: batch_time_music_test,
                                                       y_: batched_time_labels_test})
                                / batch_time_music_test.shape[0])
            Yhattestbase = sess.run(y, feed_dict={x: batch_time_music_test})
            yflat = batched_time_labels_test.flatten()
            yhatflat = Yhattestbase.flatten()
            average_precision.append(average_precision_score(yflat,
                                                             yhatflat))
            # debug_here()

        if i % 10000 == 0:
            end = time()
            print(i, '\t', round(square_error[-1], 8),
                     '\t', round(average_precision[-1], 8),
                     '\t', round(end-start, 8))
            start = time()
        batch_time_music, batched_time_labels = \
            get_batch(train_data, train_ids, batch_size)
        # debug_here()
        loss, out_net, out_gt, _ = sess.run([L, y, y_, training_step],
                                            feed_dict={x: batch_time_music,
                                                       y_: batched_time_labels})
        # debug_here()
        print(i, loss)
