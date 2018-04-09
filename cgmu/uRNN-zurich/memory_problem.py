import cPickle
import gzip
import theano
import pdb
from fftconv import cufft, cuifft
import numpy as np
import theano.tensor as T
from theano import printing
from theano.ifelse import ifelse
from models import *
from optimizations import *    
import argparse, timeit

from time import time

def generate_data(time_steps, n_data, n_sequence):
    seq = np.random.randint(1, high=9, size=(n_data, n_sequence))
    # STEPH: the sequence to remember
    #   uncertain why training examples are now dimension 0, but OK it gets
    #   transposed at the end... (probably to make concatenating easier?)
    zeros1 = np.zeros((n_data, time_steps-1))
    # STEPH: T-1 zeros
    zeros2 = np.zeros((n_data, time_steps))
    # STEPH: T zeros
    marker = 9 * np.ones((n_data, 1))
    # STEPH: 1 set of 9s ('start reproducing sequence' marker)
    zeros3 = np.zeros((n_data, n_sequence))
    # STEPH: length-of-sequence set of zeros

    x = np.concatenate((seq, zeros1, marker, zeros3), axis=1).astype('int32')
    # STEPH: the full input is: sequence, T-1 zeros, special marker,
    #   sequence-length zeros (empty category)
    y = np.concatenate((zeros3, zeros2, seq), axis=1).astype('int32')
    # STEPH: desired output is: T + length-of-seq sequence zeros, then sequence

    return x.T, y.T


def main(n_iter, n_batch, n_hidden, time_steps, learning_rate, savefile, model, input_type, out_every_t, loss_function, input_path):

    # --- Set data params ----------------
    n_input = 10
    n_output = 9
    n_sequence = 10
    n_train = int(1e5)
    n_test = int(1e4)
    num_batches = int(n_train / n_batch)

    # --- Create data --------------------
    if input_path:
        print 'Loading data from', input_path
        load_dict = cPickle.load(open(input_path, 'rb'))
        train = load_dict['train']
        train_x, train_y = train.x, train.y
        # the input data is one-hot, need to convert
        # axes: input seq, batch, one-hot-encoding
        train_x = np.int32(np.argmax(train_x, axis=2)).T
        train_y = np.int32(train_y).T
        # to compare, actually use validation data
        vali = load_dict['vali']
        test_x, test_y = vali.x, vali.y
        test_x = np.int32(np.argmax(test_x, axis=2)).T
        test_y = np.int32(test_y).T
    else:
        train_x, train_y = generate_data(time_steps, n_train, n_sequence)
        test_x, test_y = generate_data(time_steps, n_test, n_sequence)

    s_train_x = theano.shared(train_x)
    s_train_y = theano.shared(train_y)

    s_test_x = theano.shared(test_x)
    s_test_y = theano.shared(test_y)

    
    # --- Create theano graph and compute gradients ----------------------

    gradient_clipping = np.float32(1)

    if (model == 'LSTM'):           
        inputs, parameters, costs = LSTM(n_input, n_hidden, n_output, input_type=input_type,
                                         out_every_t=out_every_t, loss_function=loss_function)
        gradients = T.grad(costs[0], parameters)
        gradients = [T.clip(g, -gradient_clipping, gradient_clipping) for g in gradients]

    elif (model == 'complex_RNN'):
        inputs, parameters, costs = complex_RNN(n_input, n_hidden, n_output, input_type=input_type,
                                                out_every_t=out_every_t, loss_function=loss_function)
        gradients = T.grad(costs[0], parameters)

    elif (model == 'orthogonal_RNN'):
        # TODO: (STEPH) write orthogonal_RNN
        #    note: may want another option specifying the basis of the Lie algebra
        inputs, parameters, costs = orthogonal_RNN(n_input, n_hidden, n_output,
                                                   input_type=input_type,
                                                   out_every_t=out_every_t,
                                                   loss_function=loss_function)
        gradients = T.grad(costs[0], parameters)

    elif (model == 'general_unitary_RNN'):
        # TODO: (STEPH) write general_unitary_RNN
        #   note: may want another option specifying the basis of the Lie algebra
        inputs, parameters, costs = general_unitary_RNN(n_input, n_hidden, n_output, input_type=input_type,
                                                out_every_t=out_every_t, loss_function=loss_function)
        gradients = T.grad(costs[0], parameters)

    elif (model == 'IRNN'):
        inputs, parameters, costs = IRNN(n_input, n_hidden, n_output, input_type=input_type,
                                         out_every_t=out_every_t, loss_function=loss_function)
        gradients = T.grad(costs[0], parameters)
        gradients = [T.clip(g, -gradient_clipping, gradient_clipping) for g in gradients]

    elif (model == 'RNN'):
        inputs, parameters, costs = tanhRNN(n_input, n_hidden, n_output, input_type=input_type,
                                            out_every_t=out_every_t, loss_function=loss_function)
        gradients = T.grad(costs[0], parameters)
        gradients = [T.clip(g, -gradient_clipping, gradient_clipping) for g in gradients]
    
    else:
        print "Unsupported model:", model
        return

 
    # --- Compile theano functions --------------------------------------------------

    index = T.iscalar('i')

    updates, rmsprop = rms_prop(learning_rate, parameters, gradients)

    givens = {inputs[0] : s_train_x[:, n_batch * index : n_batch * (index + 1)],
              inputs[1] : s_train_y[:, n_batch * index : n_batch * (index + 1)]}

    givens_test = {inputs[0] : s_test_x,
                   inputs[1] : s_test_y}
    
    train = theano.function([index], [costs[0]], givens=givens, updates=updates)
    test = theano.function([], [costs[0], costs[1]], givens=givens_test)

    # --- prepare logging --- #
    train_trace_file = open(savefile + '.train.txt', 'w')
    train_trace_file.write('epoch batch train_cost\n')
    vali_trace_file = open(savefile + '.vali.txt', 'w')
    vali_trace_file.write('epoch batch vali_cost\n')
    timing_file = open(savefile + '.timing.txt', 'w')
    timing_file.write('epoch batch time\n')

    # --- Training Loop ---------------------------------------------------------------

    train_loss = []
    test_loss = []
    test_acc = []
    best_params = [p.get_value() for p in parameters]
    best_rms = [r.get_value() for r in rmsprop]
    best_test_loss = 1e6
    for i in xrange(n_iter):
        if (n_iter % num_batches == 0):
            # STEPH: this is probably supposed to be: i % num_batches == 0 ... 
            inds = np.random.permutation(n_train)
            data_x = s_train_x.get_value()
            s_train_x.set_value(data_x[:,inds])
            data_y = s_train_y.get_value()
            s_train_y.set_value(data_y[:,inds])

        train_ce = train(i % num_batches)
        # STEPH: reporting cross-entropy and not MSE, this time (in theory)
        #   the loss function isn't actually specified explicitly here, but
        #   it is presumably cross entropy...
        train_loss.append(train_ce)
        print "Iteration:", i, "(memory) (T =", time_steps, ")"
        print "cross entropy:", train_ce
        print

        if (i % 50==0):
            test_ce, acc = test()
            print
            print "TEST"
            print "cross entropy:", test_ce
            print 
            test_loss.append(test_ce)
            test_acc.append(acc)

            if test_ce < best_test_loss:
                best_params = [p.get_value() for p in parameters]
                best_rms = [r.get_value() for r in rmsprop]
                best_test_loss = test_ce

            ### writing to file ###
            # (copying naming scheme from tf version)
            batch_index = i % int(num_batches)
            epoch = i / int(num_batches)
            train_cost = train_ce
            vali_cost = test_ce
            train_trace_file.write(str(epoch) + ' ' + str(batch_index) + ' ' + str(train_cost) + '\n')
            vali_trace_file.write(str(epoch) + ' ' + str(batch_index) + ' ' + str(vali_cost) + '\n')
            train_trace_file.flush()
            vali_trace_file.flush()
            # timing stuff
            if batch_index == 0:
                dt = 0.0
                t_prev = time()
            t = time()
            dt = t - t_prev
            timing_file.write(str(epoch) + ' ' + str(batch_index) + ' ' + str(dt) + '\n')
            timing_file.flush()
            t_prev = t

            # i will retire this eventually #
            save_vals = {'parameters': [p.get_value() for p in parameters],
                         'rmsprop': [r.get_value() for r in rmsprop],
                         'train_loss': train_loss,
                         'test_loss': test_loss,
                         'test_acc': test_acc,
                         'best_params': best_params,
                         'best_rms': best_rms,
                         'best_test_loss': best_test_loss,
                         'model': model,
                         'time_steps': time_steps}

            cPickle.dump(save_vals,
                         file(savefile, 'wb'),
                         cPickle.HIGHEST_PROTOCOL)

            
if __name__=="__main__":
    parser = argparse.ArgumentParser(
        description="training a model")
    parser.add_argument("--n_iter", type=int, default=20000)
    parser.add_argument("--n_batch", type=int, default=20)
    parser.add_argument("--n_hidden", type=int, default=128)
    parser.add_argument("--time_steps", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--savefile", default='memory100_tf_v2')
    parser.add_argument("--model", default='complex_RNN')
    parser.add_argument("--input_type", default='categorical')
    parser.add_argument("--out_every_t", default='True')
    parser.add_argument("--loss_function", default='CE')
    parser.add_argument("--input_path", default="")

    args = parser.parse_args()
    dict = vars(args)
    
    # same data as I'm using in the TF experiments...
    if dict['time_steps'] == 100:
        dict['input_path'] = '/home/hyland/git/complex_RNN/tf/input/memory/1470766867_100.pk'
    elif dict['time_steps'] == 200:
        dict['input_path'] = '/home/hyland/git/complex_RNN/tf/input/memory/1470767064_200.pk'
    elif dict['time_steps'] == 300:
        dict['input_path'] = '/home/hyland/git/complex_RNN/tf/input/memory/1470767409_300.pk'
    elif dict['time_steps'] == 500:
        dict['input_path'] = '/home/hyland/git/complex_RNN/tf/input/memory/1470767936_500.pk'
    else:
        raise ValueError(dict['time_steps'])
    
    kwargs = {'n_iter': dict['n_iter'],
              'n_batch': dict['n_batch'],
              'n_hidden': dict['n_hidden'],
              'time_steps': dict['time_steps'],
              'learning_rate': np.float32(dict['learning_rate']),
              'savefile': dict['savefile'],
              'model': dict['model'],
              'input_type': dict['input_type'],
              'out_every_t': 'True'==dict['out_every_t'],
              'loss_function': dict['loss_function'],
              'input_path': dict['input_path']}

    # save options
    options_file = open(dict['savefile'] + '.options.txt', 'w')
    for (key, val) in kwargs.iteritems():
        options_file.write(key + ' ' + str(val) + '\n')
    options_file.close()
        
    # STEPH: since this is _memory problem_, only some settings are allowed!
    # ( I could probably enforce this during parsing, too )
    ERR = False
    if not kwargs['loss_function'] == 'CE':
        print 'loss function must be CE'
        ERR = True
    if not kwargs['input_type'] == 'categorical':
        print 'input_type must be categorical'
        ERR = True
    if not kwargs['out_every_t'] == True:
        print 'out_every_t must be True'
        ERR = True
    if ERR:
        sys.exit('Arguments failed checks, quitting.')
    else:
        main(**kwargs)
