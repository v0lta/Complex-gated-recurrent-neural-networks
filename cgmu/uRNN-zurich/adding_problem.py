import cPickle
import gzip
import theano
from fftconv import cufft, cuifft
import numpy as np
import theano.tensor as T
from theano.ifelse import ifelse
from models import *
from optimizations import *    
import argparse
import pdb

import cPickle
import data

def generate_data(time_steps, n_data):
    # STEPH: n_data is n_train or n_test
    #   time_steps is the length of the sequence, I think
    #   that is to say, the size of a single training instance
    x = np.asarray(np.zeros((time_steps, int(n_data), 2)),
                   dtype=theano.config.floatX)

    x[:, :, 0] = np.asarray(np.random.uniform(low=0.,
                                              high=1.,
                                              size=(time_steps, n_data)),
                            dtype=theano.config.floatX)

    inds = np.asarray(np.random.randint(time_steps/2, size=(n_data, 2)))
    inds[:, 1] += time_steps / 2  
    # STEPH: [:, 0] is from 0 til time_steps/2, [:, 1] is [:, 0] + time_steps/2
    #   basically, these just pick out which two elements will be added together
    
    for i in range(int(n_data)):
        x[inds[i, 0], i, 1] = 1.0
        x[inds[i, 1], i, 1] = 1.0
    # STEPH: x[:, :, 1] is 1 in the row of given by the relevant n_data of inds
    #   x[:, :, 1] is otherwise all 0s
 
    y = (x[:,:,0] * x[:,:,1]).sum(axis=0)
    # STEPH: before summing, y would be shape: (time_steps, n_data)...
    #   then we sum over time_steps, so this is just n_data length
    y = np.reshape(y, (n_data, 1))
    # STEPH: now its shape is (n_data, 1)
    #   this is: for each example in n_data, it's the sum of two elements from
    #   the training instance (size time_steps)...

    return x, y

    
    
def main(n_iter, n_batch, n_hidden, time_steps, learning_rate, savefile, 
         model, input_type, out_every_t, loss_function, input_path):
    
    # --- Set data params ----------------
    n_input = 2
    n_output = 1
    n_train = 1e5
    n_test = 1e4
    num_batches = n_train / n_batch
  

    # --- Create data --------------------
    # STEPH: actually, load data
    if input_path:
        print 'Loading data from', input_path 
        load_dict = cPickle.load(open(input_path, 'rb'))
        train = load_dict['train']
        train_x, train_y = train.x, train.y
        # fix the axes
        train_x = np.swapaxes(train_x, 0, 1)
        # to compare, actually use validation data
        vali = load_dict['vali']
        test_x, test_y = vali.x, vali.y
        test_x = np.swapaxes(test_x, 0, 1)
    else:
        train_x, train_y = generate_data(time_steps, n_train)
        test_x, test_y = generate_data(time_steps, n_test)

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
        #   note: may want another option specifying the basis of the Lie algebra
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
    # STEPH: rms_prop is in optimizations.py
    
    givens = {inputs[0] : s_train_x[:, n_batch * index : n_batch * (index + 1), :],
              inputs[1] : s_train_y[n_batch * index : n_batch * (index + 1), :]}
    # STEPH: pick out the batch of examples (values in x and labels in y)
    
    givens_test = {inputs[0] : s_test_x,
                   inputs[1] : s_test_y}
    # STEPH: the same test regardless of batch
    
    train = theano.function([index], costs[0], givens=givens, updates=updates)
    # STEPH: given an index, this outputs costs[0], 
    #   givens are specific substitutions to make in the computation graph
    #   updates are expressions for new SharedVariable values (parameters)

    test = theano.function([], costs[0], givens=givens_test)
    # STEPH: no updates, here

    # YOLO
    print 'pretrain...'
    print updates
    # DEYOLO

    # --- prepare logging... ---#
    train_trace_file = open(savefile + '.train.txt', 'w')
    train_trace_file.write('epoch batch train_cost\n')
    vali_trace_file = open(savefile + '.vali.txt', 'w')
    vali_trace_file.write('epoch batch vali_cost\n')

    # --- Training Loop ---------------------------------------------------------------

    train_loss = []
    test_loss = []
    best_params = [p.get_value() for p in parameters]
    best_rms = [r.get_value() for r in rmsprop]                
    best_test_loss = 1e6
    for i in xrange(n_iter):
        if (n_iter % int(num_batches) == 0):
            inds = np.random.permutation(int(n_train))
            data_x = s_train_x.get_value()
            s_train_x.set_value(data_x[:,inds,:])
            data_y = s_train_y.get_value()
            s_train_y.set_value(data_y[inds,:])

        # YOLO
#        for p in parameters:
#            print p.name
#            print p.get_value()
        # DEYOLO

        train_mse = train(i % int(num_batches))
        # STEPH: remember, input of train is the batch number, 
        #   and output is costs[0]
        train_loss.append(train_mse)
        print "Iteration:", i, "(adding) (T =", time_steps, ")"
        print "mse:", train_mse
        print

        if (i % 50==0):
            test_mse = test()
            # STEPH: test takes no inputs, as it is a fixed test set
            print
            print "TEST"
            print "mse:", test_mse
            print 
            test_loss.append(test_mse)

            if test_mse < best_test_loss:
                best_params = [p.get_value() for p in parameters]
                best_rms = [r.get_value() for r in rmsprop]                
                best_test_loss = test_mse

           
            ### writing to file ###
            # (copying naming scheme from tf version)
            batch_index = i % int(num_batches)
            epoch = i / int(num_batches)
            train_cost = train_mse
            vali_cost = test_mse
            train_trace_file.write(str(epoch) + ' ' + str(batch_index) + ' ' + str(train_cost) + '\n')
            vali_trace_file.write(str(epoch) + ' ' + str(batch_index) + ' ' + str(vali_cost) + '\n')
            train_trace_file.flush()
            vali_trace_file.flush()
    
            # i will retire this eventually #
            save_vals = {'parameters': [p.get_value() for p in parameters],
                         'rmsprop': [r.get_value() for r in rmsprop],
                         'train_loss': train_loss,
                         'test_loss': test_loss,
                         'best_params': best_params,
                         'best_rms': best_rms,
                         'best_test_loss': best_test_loss,
                         'model': model,
                         'time_steps': time_steps}

            cPickle.dump(save_vals,
                         file(savefile, 'wb'),
                         cPickle.HIGHEST_PROTOCOL)
            # STEPH: HIGHEST_PROTOCOL is highest protocol version available !!!

        

    
if __name__=="__main__":
    parser = argparse.ArgumentParser(
        description="training a model")
    parser.add_argument("--n_iter", type=int, default=40000)
    #parser.add_argument("--n_iter", type=int, default=20000)
    parser.add_argument("--n_batch", type=int, default=20)
    parser.add_argument("--n_hidden", type=int, default=512)
    #parser.add_argument("--n_hidden", type=int, default=80)
    parser.add_argument("--time_steps", type=int, default=750)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--savefile", default='addingT750_tf_v2')
    parser.add_argument("--model", default='complex_RNN')
    parser.add_argument("--input_type", default='real')
    parser.add_argument("--out_every_t", default='False')
    parser.add_argument("--loss_function", default='MSE')
    parser.add_argument("--input_path", default="")

    args = parser.parse_args()
    dict = vars(args)
    # STEPH: argh bad naming here, dict is a built-in name in python !!

    # same data as I'm using in the TF experiments...
    if dict['time_steps'] == 100:
        dict['input_path'] = '/home/hyland/git/complex_RNN/tf/input/adding/1470744790_100.pk'
    elif dict['time_steps'] == 200:
        dict['input_path'] = '/home/hyland/git/complex_RNN/tf/input/adding/1470744860_200.pk'
    elif dict['time_steps'] == 400:
        dict['input_path'] = '/home/hyland/git/complex_RNN/tf/input/adding/1470744994_400.pk'
    elif dict['time_steps'] == 750:
        dict['input_path'] = '/home/hyland/git/complex_RNN/tf/input/adding/1470745056_750.pk'
    else:
        raise ValueError(dict['time_steps'])

    ### debug
#    dict['input_path'] = ''

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

    # STEPH:
    # # force sanity on the arguments
    ERR = False
    if not kwargs['loss_function'] == 'MSE':
        print 'loss function must be MSE'
        ERR = True
    if not kwargs['input_type'] == 'real':
        print 'input_type must be real'
        ERR = True
    if not kwargs['out_every_t'] == False:
        print 'out_every_t must be False'
        ERR = True
    if ERR:
        sys.exit('Arguments failed checks, quitting.')
    else:
        main(**kwargs)
