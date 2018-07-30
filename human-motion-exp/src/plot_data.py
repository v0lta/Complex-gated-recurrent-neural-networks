import data_utils
import numpy as np
import scipy as sci
import matplotlib.pyplot as plt

from forward_kinematics import _some_variables, fkl, revert_coordinate_space

from IPython.core.debugger import Tracer; debug_here = Tracer()


act_lst = ['walking', 'eating', 'smoking', 'discussion', 
           'directions', 'greeting', 'phoning', 'posing',
           'purchases', 'sitting', 'sittingdown', 'takingphoto',
           'waiting', 'walkingdog', 'walkingtogether']

def read_all_data(actions=act_lst, seq_length_in=50, seq_length_out=25, 
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
encoder_inputs, decoder_inputs, decoder_outputs = get_batch(train_set)
expmap_gt = train_set[list(train_set.keys())[0]]

plt.imshow(encoder_inputs[0, :, :]); plt.show()

batch_size, nframes, dims = encoder_inputs.shape

xyz_gt = np.zeros((nframes, 96))
for i in range(nframes):
    xyz_gt[i,:] = fkl(expmap_gt[i,:], parent, 
                      offset, rotInd, expmapInd)


# === Plot and animate ===
fig = plt.figure()
ax = plt.gca(projection='3d')
ob = viz.Ax3DPose(ax)

# Plot the conditioning ground truth
for i in range(nframes):
    ob.update( xyz_gt[i,:] )
    plt.show(block=False)
    fig.canvas.draw()
    plt.pause(0.01)