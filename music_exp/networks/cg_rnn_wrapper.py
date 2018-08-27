import tensorflow as tf


class cgRNNWrapper(tf.nn.rnn_cell.RNNCell):

    def __init__(self, cell, net_out_prob):
        self._cell = cell
        self.net_out_prob = net_out_prob

    def select_out_or_target(self, groundtruth, net_output):
        """ Select the last system output value, or the ground-truth.
            The probability of picking the ground truth value is
            given by self.net_out_prob value. """

        def pick_ground_truth():
            """ Return the true value, from the set annotations."""
            return groundtruth

        def pick_last_output():
            """ Return the last network output (sometimes wrong),
                to expose the network to false predictions. """
            return net_output

        def pred():
            """ Return the network output net_out_prob*100 percent
                of the time."""
            return tf.greater(rand_val, self.net_out_prob)

        rand_val = tf.random_uniform(shape=(), minval=0.0, maxval=1.0,
                                     dtype=tf.float32, seed=None,
                                     name='random_uniform')
        net_output = tf.cond(pred(),
                             pick_ground_truth,
                             pick_last_output,
                             name='truth_or_output_sel')
        return net_output


    def __call__(self, inputs, state):

        net_out = state[0]
        
        net_in = self.select_out_or_target(inputs, )
