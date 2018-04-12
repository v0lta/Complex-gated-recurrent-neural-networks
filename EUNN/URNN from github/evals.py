import tensorflow as tf
from problems.adding_problem import AddingProblemDataset
from problems.copying_memory_problem import CopyingMemoryProblemDataset
from networks.tf_rnn import TFRNN
from networks.urnn_cell import URNNCell
import numpy as np

'''
        name,
        rnn_cell,
        num_in,
        num_hidden, 
        num_out,
        num_target,
        single_output,
        activation_hidden,
        activation_out,
        optimizer,
        loss_function):
'''

loss_path='results/'

glob_learning_rate = 0.001
glob_decay = 0.9

def baseline_cm(timesteps):
    return 10*np.log(8) / timesteps

def baseline_ap():
    return 0.167

def serialize_loss(loss, name):
    file=open(loss_path + name, 'w')
    for l in loss:
        file.write("{0}\n".format(l))

class Main:
    def init_data(self):
        print('Generating data...')

        # init copying memory problem
        self.cm_batch_size=50
        self.cm_epochs=10

        self.cm_timesteps=[120, 220, 320, 520]
        self.cm_samples=100000
        self.cm_data=[CopyingMemoryProblemDataset(self.cm_samples, timesteps) for timesteps in self.cm_timesteps]
        self.dummy_cm_data=CopyingMemoryProblemDataset(100, 50) # samples, timestamps

        # init adding problem
        self.ap_batch_size=50
        self.ap_epochs=10

        self.ap_timesteps=[100, 200, 400, 750]
        self.ap_samples=[30000, 50000, 40000, 100000]
        self.ap_data=[AddingProblemDataset(sample, timesteps) for 
                      timesteps, sample in zip(self.ap_timesteps, self.ap_samples)]
        self.dummy_ap_data=AddingProblemDataset(100, 50) # samples, timestamps

        print('Done.')

    def train_network(self, net, dataset, batch_size, epochs):
        sample_len = str(dataset.get_sample_len())
        print('Training network ', net.name, '... timesteps=',sample_len)
        net.train(dataset, batch_size, epochs)
        # loss_list has one number for each batch (step)
        serialize_loss(net.get_loss_list(), net.name + sample_len)
        print('Training network ', net.name, ' done.')

    def train_urnn_for_timestep_idx(self, idx):
        print('Initializing and training URNNs for one timestep...')

        # CM

        tf.reset_default_graph()
        self.cm_urnn=TFRNN(
            name="cm_urnn",
            num_in=1,
            num_hidden=128,
            num_out=10,
            num_target=1,
            single_output=False,
            rnn_cell=URNNCell,
            activation_hidden=None, # modReLU
            activation_out=tf.identity,
            optimizer=tf.train.RMSPropOptimizer(learning_rate=glob_learning_rate, decay=glob_decay),
            loss_function=tf.nn.sparse_softmax_cross_entropy_with_logits)
        self.train_network(self.cm_urnn, self.cm_data[idx], 
                           self.cm_batch_size, self.cm_epochs)

        # AP

        tf.reset_default_graph()
        self.ap_urnn=TFRNN(
            name="ap_urnn",
            num_in=2,
            num_hidden=512,
            num_out=1,
            num_target=1,
            single_output=True,
            rnn_cell=URNNCell,
            activation_hidden=None, # modReLU
            activation_out=tf.identity,
            optimizer=tf.train.RMSPropOptimizer(learning_rate=glob_learning_rate, decay=glob_decay),
            loss_function=tf.squared_difference)
        self.train_network(self.ap_urnn, self.ap_data[idx], 
                           self.ap_batch_size, self.ap_epochs)

        print('Init and training URNNs for one timestep done.')


    def train_rnn_lstm_for_timestep_idx(self, idx):
        print('Initializing and training RNN&LSTM for one timestep...')

        # CM

        tf.reset_default_graph()
        self.cm_simple_rnn=TFRNN(
            name="cm_simple_rnn",
            num_in=1,
            num_hidden=80,
            num_out=10,
            num_target=1,
            single_output=False,
            rnn_cell=tf.contrib.rnn.BasicRNNCell,
            activation_hidden=tf.tanh,
            activation_out=tf.identity,
            optimizer=tf.train.RMSPropOptimizer(learning_rate=glob_learning_rate, decay=glob_decay),
            loss_function=tf.nn.sparse_softmax_cross_entropy_with_logits)
        self.train_network(self.cm_simple_rnn, self.cm_data[idx], 
                           self.cm_batch_size, self.cm_epochs)

        tf.reset_default_graph()
        self.cm_lstm=TFRNN(
            name="cm_lstm",
            num_in=1,
            num_hidden=40,
            num_out=10,
            num_target=1,
            single_output=False,
            rnn_cell=tf.contrib.rnn.LSTMCell,
            activation_hidden=tf.tanh,
            activation_out=tf.identity,
            optimizer=tf.train.RMSPropOptimizer(learning_rate=glob_learning_rate, decay=glob_decay),
            loss_function=tf.nn.sparse_softmax_cross_entropy_with_logits)
        self.train_network(self.cm_lstm, self.cm_data[idx], 
                           self.cm_batch_size, self.cm_epochs)

        # AP

        tf.reset_default_graph()
        self.ap_simple_rnn=TFRNN(
            name="ap_simple_rnn",
            num_in=2,
            num_hidden=128,
            num_out=1,
            num_target=1,
            single_output=True,
            rnn_cell=tf.contrib.rnn.BasicRNNCell,
            activation_hidden=tf.tanh,
            activation_out=tf.identity,
            optimizer=tf.train.RMSPropOptimizer(learning_rate=glob_learning_rate, decay=glob_decay),
            loss_function=tf.squared_difference)
        self.train_network(self.ap_simple_rnn, self.ap_data[idx], 
                           self.ap_batch_size, self.ap_epochs)

        tf.reset_default_graph()
        self.ap_lstm=TFRNN(
            name="ap_lstm",
            num_in=2,
            num_hidden=128,
            num_out=1,
            num_target=1,
            single_output=True,
            rnn_cell=tf.contrib.rnn.LSTMCell,
            activation_hidden=tf.tanh,
            activation_out=tf.identity,
            optimizer=tf.train.RMSPropOptimizer(learning_rate=glob_learning_rate, decay=glob_decay),
            loss_function=tf.squared_difference)
        self.train_network(self.ap_lstm, self.ap_data[idx], 
                           self.ap_batch_size, self.ap_epochs)

        print('Init and training networks for one timestep done.')

    def train_networks(self):
        print('Starting training...')

        timesteps_idx=4
        for i in range(timesteps_idx):
            main.train_urnn_for_timestep_idx(i)
        for i in range(timesteps_idx):
            main.train_rnn_lstm_for_timestep_idx(i)

        print('Done and done.')

main=Main()
main.init_data()
main.train_networks()
