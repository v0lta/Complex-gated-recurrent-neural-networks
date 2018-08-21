import tensorflow as tf

from tensorflow.contrib.rnn.python.ops.core_rnn_cell import RNNCell
import custom_cells as cc


# todo implement single and multilayer output resampling.
def ResampleWrapper(RNNCell):

    def __init__(self, cell, net_out_prob, layer_no=1):
        """Create a cell with added residual connection.

        Args:
          cell: an RNNCell. The input is added to the output.
          net_out_prob: RNN-Resample probability.
        Raises:
          TypeError: if cell is not an RNNCell.
        """
        if not isinstance(cell, RNNCell):
            raise TypeError("The parameter cell is not a RNNCell.")

        self._cell = cell
        self._net_out_prob = net_out_prob
        self._layer_no = layer_no

    def select_out_or_target(self, groundtruth, output):
        """ Select the last system output value, or the ground-truth.
            The probability of picking the ground truth value is
            given by self.net_out_prob value. """

        def pick_ground_truth():
            """ Return the true value, from the set annotations."""
            return groundtruth

        def pick_last_output():
            """ Return the last network output (sometimes wrong),
                to expose the network to false predictions. """
            return output

        def pred():
            """ Return the network output net_out_prob*100 percent
                of the time."""
            return tf.greater(rand_val, self.net_out_prob)

        rand_val = tf.random_uniform(shape=(), minval=0.0, maxval=1.0,
                                     dtype=tf.float32, seed=None,
                                     name='random_uniform')
        to_net = tf.cond(pred(),
                         pick_ground_truth,
                         pick_last_output,
                         name='truth_or_output_sel')
        return to_net

    def call(self, inputs, state, ground_truth, scope=None):
        """Run the cell and concatenate the sampled output."""
        prev_out = state[0]
        to_input = self.select_out_or_target(ground_truth, prev_out)

        if inputs.dtype == tf.complex64 or inputs.dtype == tf.complex128:
            to_input = tf.complex(inputs, tf.zeros_like(inputs))

        inputs = tf.concatenate([inputs, to_input], axis=-1)

        for layer in range(self._layer_no):
            output, new_state = self._cell(inputs, state, scope)

        return output, new_state
