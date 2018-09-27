import data_utils
import numpy as np
import scipy as sci
import tensorflow as tf
tf.enable_eager_execution()
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from forward_kinematics import _some_variables, fkl, revert_coordinate_space

from IPython.core.debugger import Tracer; debug_here = Tracer()


act_lst = ['walking', 'eating', 'smoking', 'discussion', 
           'directions', 'greeting', 'phoning', 'posing',
           'purchases', 'sitting', 'sittingdown', 'takingphoto',
           'waiting', 'walkingdog', 'walkingtogether']
walking_lst = ['walking']


def read_all_data(actions=walking_lst, seq_length_in=50, seq_length_out=25, 
                  data_dir="./data/h3.6m/dataset", one_hot=True):
  """
  Loads data for training/testing and normalizes it.

  Args
    actions: list of strings (actions) to load
    seq_length_in: number of frames to use in the burn-in sequence
    seq_length_out: number of frames to use in the output sequence
    data_dir: directory to load the data from
    one_hot: whether to use one-hot encoding per action
  Returns
    train_set: dictionary with normalized training data
    test_set: dictionary with test data
    data_mean: d-long vector with the mean of the training data
    data_std: d-long vector with the standard dev of the training data
    dim_to_ignore: dimensions that are not used becaused stdev is too small
    dim_to_use: dimensions that we are actually using in the model
  """

  # === Read training data ===
  print ("Reading training data (seq_len_in: {0}, seq_len_out {1}).".format(
           seq_length_in, seq_length_out))

  train_subject_ids = [1,6,7,8,9,11]
  test_subject_ids = [5]

  train_set, complete_train = data_utils.load_data( data_dir, train_subject_ids, actions, one_hot )
  test_set,  complete_test  = data_utils.load_data( data_dir, test_subject_ids,  actions, one_hot )

  # Compute normalization stats
  data_mean, data_std, dim_to_ignore, dim_to_use = data_utils.normalization_stats(complete_train)

  # Normalize -- subtract mean, divide by stdev
  train_set = data_utils.normalize_data( train_set, data_mean, data_std, dim_to_use, actions, one_hot )
  test_set  = data_utils.normalize_data( test_set,  data_mean, data_std, dim_to_use, actions, one_hot )
  print("done reading data.")

  return train_set, test_set, data_mean, data_std, dim_to_ignore, dim_to_use


def get_batch(data):
    """Get a random batch of data from the specified bucket, prepare for step.
    Args
      data: a list of sequences of size n-by-d to fit the model to.
      actions: a list of the actions we are using
    Returns
      The tuple (encoder_inputs, decoder_inputs, decoder_outputs);
      the constructed batches have the proper format to call step(...) later.
    """
    batch_size = 1
    source_seq_len = 50
    target_seq_len = 25
    input_size = 25
    HUMAN_SIZE = 54
    input_size = HUMAN_SIZE + len(act_lst)

    # Select entries at random
    all_keys    = list(data.keys())
    chosen_keys = np.random.choice( len(all_keys), batch_size )

    # How many frames in total do we need?
    total_frames = source_seq_len + target_seq_len

    encoder_inputs  = np.zeros((batch_size, source_seq_len-1, input_size), dtype=float)
    decoder_inputs  = np.zeros((batch_size, target_seq_len, input_size), dtype=float)
    decoder_outputs = np.zeros((batch_size, target_seq_len, input_size), dtype=float)

    for i in range( batch_size ):
        the_key = all_keys[ chosen_keys[i] ]

        # Get the number of frames
        n, _ = data[ the_key ].shape

        # Sample somewherein the middle
        idx = np.random.randint( 16, n-total_frames )

        # Select the data around the sampled points
        data_sel = data[ the_key ][idx:idx+total_frames ,:]

        # Add the data
        encoder_inputs[i,:,0:input_size]  = data_sel[0:source_seq_len-1, :]
        decoder_inputs[i,:,0:input_size]  = data_sel[source_seq_len-1:source_seq_len+target_seq_len-1, :]
        decoder_outputs[i,:,0:input_size] = data_sel[source_seq_len:, 0:input_size]
    return encoder_inputs, decoder_inputs, decoder_outputs


parent, offset, rotInd, expmapInd = _some_variables()
train_set, test_set, data_mean, data_std, dim_to_ignore, dim_to_use = read_all_data()

data = train_set[(1, 'walking', 1, 'even')][:500, :]

if 0:
    # naive approach.
    
    plt.imshow(np.log(np.abs(np.fft.rfft(data.transpose()))[:-2, :-2]));
    plt.show()
    # debug_here()

    # plot the fourier domain data.
    time = test_data.shape[0]
    window_size = 50

    fig = plt.figure()
    im_lst = []
    for i in range(0, time // window_size - 1):
        start = i * window_size
        end = (i+1) * window_size
        # print(start, end)
        current_data = test_data[start:end, :]
        frame = np.abs(np.fft.rfft(current_data.transpose()))
        im = plt.imshow(frame, animated=True)
        im_lst.append([im])

    ani = animation.ArtistAnimation(fig, im_lst, interval=250, repeat=False)
    plt.show()

if 1:
    # do this the windowed way.
    frame_length = 60
    frame_step = 10

    center = True
    if center:
        pad_amount = 2 * (frame_length - frame_step)
        x_pad = tf.pad(data.astype(np.float32).transpose(),
                       [[0, 0], [pad_amount // 2, pad_amount // 2]], 'REFLECT')
    else:
        x_pad = x.astype(np.float32).transpose()

    # f = tf.contrib.signal.frame(x_pad, frame_length, frame_step, pad_end=False)
    # w = tf.contrib.signal.hann_window(frame_length, periodic=True)
    # stfts = tf.spectral.rfft(f * w, fft_length=[frame_length])
    # stfts = tf.spectral.rfft(f)
    stfts = tf.contrib.signal.stft(x_pad, frame_length, frame_step)

    # real_frames = tf.spectral.irfft(stfts)
    # denom = tf.square(w)
    # overlaps = -(-frame_length // frame_step)
    # denom = tf.pad(denom, [(0, overlaps * frame_step - frame_length)])
    # denom = tf.reshape(denom, [overlaps, frame_step])
    # denom = tf.reduce_sum(denom, 0, keepdims=True)
    # denom = tf.tile(denom, [overlaps, 1])
    # denom = tf.reshape(denom, [overlaps * frame_step])
    # w_inv = w / (denom)
    # real_frames = real_frames*w_inv

    # if center and pad_amount > 0:
    #     real_frames = real_frames[pad_amount // 2:-pad_amount // 2]
    output_T = tf.contrib.signal.inverse_stft(
        stfts, frame_length, frame_step,
        window_fn=tf.contrib.signal.inverse_stft_window_fn(frame_step))
    if center and pad_amount > 0:
        output = tf.transpose(output_T[:, pad_amount // 2:-pad_amount // 2])
    else:
        output = tf.transpose(output_T)

    output_array = np.array(output)
    print(np.linalg.norm(data.astype(np.float32) - output_array))

    frame_no = stfts.get_shape()[-2].value

    fig = plt.figure()
    im_lst = []
    for i in range(0, frame_no):
        frame = np.abs(np.array(stfts[:, i, :]))
        im = plt.imshow(frame, animated=True)
        im_lst.append([im])
    ani = animation.ArtistAnimation(fig, im_lst, interval=250, repeat=False)
    plt.show()