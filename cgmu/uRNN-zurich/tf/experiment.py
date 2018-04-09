#!/usr/bin/env ipython
#
# Outermost experiment-running script.
# aka Stephanie is still learning TensorFlow edition
# aka refactoring everything forever edition
# ... PART TWO!
#
# ------------------------------------------
# author:       Stephanie Hyland (@corcra)
# date:         22/3/16, 1/6/16
# ------------------------------------------

import tensorflow as tf
import numpy as np
import pdb
import cPickle
import argparse
from time import time
import sys

# local imports
from models import RNN
from data import ExperimentData
from unitary_np import lie_algebra_element, lie_algebra_basis_element, numgrad_lambda_update, eigtrick_lambda_update
from scipy.linalg import expm

from copy import deepcopy
import cProfile

# === some bools === #

DO_TEST = False
#SAVE_INTERNAL_GRADS = True
SAVE_INTERNAL_GRADS = False
#TIMING = False              # record time between batches
TIMING = True                # record time between batches

# === fns === #

def save_options(options):
    """ so I can stop forgetting what learning rate I used... """
    if options['identifier']:
        mname = options['identifier'] + '_' + options['model'] + '_T' + str(options['T']) + '_n' + str(options['state_size'])
    else:
        mname = options['model'] + '_T' + str(options['T']) + '_n' + str(options['state_size'])
    options_path = 'output/' + options['task'] + '/' + mname + '.options.txt'
    print 'Saving run options to', options_path
    options_file = open(options_path, 'w')
    for (key, value) in options.iteritems():
        options_file.write(key + ' ' + str(value) + '\n')
    options_file.close()
    return True

def get_cost(outputs, y, loss_type='MSE'):
    """
    Either cross-entropy or MSE.
    This will involve some averaging over a batch innit.

    Let's clarify some shapes:
        outputs is a LIST of length input_size,
            each element is a Tensor of shape (batch_size, output_size)
        y is a Tensor of shape (batch_size, output_size)
    """
    if loss_type == 'MSE':
        # mean squared error
        # discount all but the last of the outputs
        output = outputs[-1]
        # now this object is shape batch_size, output_size (= 1, it should be)
        cost = tf.reduce_mean(tf.sub(output, y) ** 2, 0)[0]
    elif loss_type == 'CE':
        # cross entropy
        # (there may be more efficient ways to do this)
        cost = tf.zeros([1])
        for (i, output) in enumerate(outputs):
            # maybe this is wrong
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(output, y[:, i])
            cost = tf.add(cost, tf.reduce_mean(cross_entropy))
        cost = tf.squeeze(tf.div(cost, i + 1))
    elif loss_type == 'mnist':
        # just use the last output (this is a column for the whole batch, remember)
        output = outputs[-1]
        # mean categorical cross entropy
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(output, y)
        cost = tf.reduce_mean(cross_entropy)
    else:
        raise NotImplementedError
#    tf.scalar_summary('cost', cost)
    return cost

# == some gradient-specific fns == #
def create_optimiser(learning_rate):
    print 'WARNING: RMSProp does not support complex variables!'
    # TODO: add complex support to RMSProp
    # decay and momentum are copied from theano version values
    opt = tf.train.RMSPropOptimizer(learning_rate=learning_rate,
                                    decay=0.9,
                                    momentum=0.0)
    return opt

def get_gradients(opt, cost, clipping=False, variables=None):
    if variables is None:
        gradient_variables = tf.trainable_variables()
    else:
        gradient_variables = variables
    g_and_v = opt.compute_gradients(cost, gradient_variables)
    print 'Calculating gradients of cost with respect to Variables:'
    for (g, v) in g_and_v:
        print v.name, v.dtype, v.get_shape()
#        if not v is None and not g is None:
#            tf.histogram_summary(v.name + 'grad', g)
#            tf.histogram_summary(v.name, v)
    if clipping:
        g_and_v = [(tf.clip_by_value(g, -1.0, 1.0), v) for (g, v) in g_and_v]
    return g_and_v

def update_variables(opt, g_and_v):
    train_opt = opt.apply_gradients(g_and_v, name='RMSProp_update')
    return train_opt
    
def assign_variable(v, newv):
    """
    Just assign a single variable.
    """
    assign_opt = v.assign(newv)
    return assign_opt
    
# do everything all at once
def update_step(cost, learning_rate, clipping=False):
    print 'WARNING: RMSProp does not support complex variables!'
    # TODO: add complex support to RMSProp
    # decay and momentum are copied from theano version values
    opt = tf.train.RMSPropOptimizer(learning_rate=learning_rate,
                                    decay=0.9,
                                    momentum=0.0)
    g_and_v = opt.compute_gradients(cost, tf.trainable_variables())
    print 'By the way, the gradients of cost',
    print 'are with respect to the following Variables:'
    for (g, v) in g_and_v:
        print v.name, v.dtype, v.get_shape()
        if not v is None and not g is None:
            tf.histogram_summary(v.name + 'grad', g)
            tf.histogram_summary(v.name, v)
    if clipping:
        g_and_v = [(tf.clip_by_value(g, -1.0, 1.0), v) for (g, v) in g_and_v]
    train_opt = opt.apply_gradients(g_and_v, name='RMSProp_update')
    return train_opt

def get_data(load_path, task, T, ntrain=int(1e5), nvali=int(1e4), ntest=int(1e4)):
    """
    Either load or generate data.
    """
    if load_path == '':
        print 'No data path provided...'
        # generate it
        if task == 'mnist':
            train = ExperimentData(ntrain, 'mnist_train', T)
            vali = ExperimentData(nvali, 'mnist_vali', T)
            test = ExperimentData(ntest, 'mnist_test', T)
            save_path = 'input/' + task + '/mnist.pk'
        elif task == 'mnist_perm':
            train = ExperimentData(ntrain, 'mnist_train', T, mnist_perm=True)
            vali = ExperimentData(nvali, 'mnist_vali', T, mnist_perm=True)
            test = ExperimentData(ntest, 'mnist_test', T, mnist_perm=True)
            save_path = 'input/' + task + '/mnist_perm.pk'
        else:
            train = ExperimentData(ntrain, task, T)
            vali = ExperimentData(nvali, task, T)
            test = ExperimentData(ntest, task, T)
            save_path = 'input/' + task + '/' + str(ntrain) + '_' + str(nvali) + '_' + str(int(time())) + '_' + str(T) + '.pk'
        print '...generating and saving to', save_path
        save_dict = {'train': train, 'vali': vali, 'test': test}
        cPickle.dump(save_dict, open(save_path, 'wb'))
    else:
        print 'Loading data from', load_path
        load_dict = cPickle.load(open(load_path, 'rb'))
        train = load_dict['train']
        vali = load_dict['vali']
        test = load_dict['test']
    return train, vali, test

# == and now for main == #
def run_experiment(task, batch_size, state_size, T, model, data_path, 
                  gradient_clipping, learning_rate, num_epochs, identifier, 
                  verbose):
    print 'running', task, 'task with', model, 'and state size', state_size
 
    # === data === #
    train_data, vali_data, test_data = get_data(data_path, task, T)
    num_batches = train_data.N / batch_size
    x, y = train_data.make_placeholders() # (doesn't actually matter which one we make placeholders out of)

    # === set up the model === #
    sequence_length = train_data.sequence_length
    input_size = train_data.input_size
    if task == 'adding':
        output_size = 1
        loss_type = 'MSE'
        assert input_size == 2
        assert sequence_length == T
    elif task == 'memory':
        output_size = 9
        loss_type = 'CE'
        assert input_size == 10
        assert sequence_length == T + 20
    elif 'mnist' in task:
        output_size = 10
        loss_type = 'mnist'
        assert input_size == 1
        assert sequence_length == 28*28

    if verbose: print 'setting up RNN...'
    if model == 'uRNN':
        # generate initial lambdas
        lambdas = np.random.normal(size=(state_size*state_size))
        # initialise with zeroes
        #lambdas = np.random.uniform(low=-1e-5, high=1e-5, size=(state_size*state_size))
        # transpose because that's how it goes in the RNN
        Uinit = expm(lie_algebra_element(state_size, lambdas)).T
        Uinit_re = np.real(Uinit)
        Uinit_im = np.imag(Uinit)
        # now create the RNN
        outputs = RNN(model, x, input_size, state_size, output_size, 
                      sequence_length=sequence_length,
                      init_re=Uinit_re, init_im=Uinit_im)
    else:
        outputs = RNN(model, x, input_size, state_size, output_size, 
                      sequence_length=sequence_length)

    # === logging === #
    if identifier:
        mname = identifier + '_' + model + '_T' + str(T) + '_n' + str(state_size)
    else:
        mname = model + '_T' + str(T) + '_n' + str(state_size)
  
    # update options with path...?# 
    options_path = 'output/' + options['task'] + '/' + mname + '.options.txt'
    
    best_model_path = 'output/' + task + '/' + mname + '.best_model.ckpt'
    best_vali_cost = 1e6
    
    vali_trace_path = 'output/' + task + '/' + mname + '.vali.txt'
    vali_trace_file = open(vali_trace_path, 'w')
    vali_trace_file.write('epoch batch vali_cost\n')
    train_trace_path = 'output/' + task + '/' + mname + '.train.txt'
    train_trace_file = open(train_trace_path, 'w')
    train_trace_file.write('epoch batch train_cost\n')
    if 'mnist' in task:
        vali_acc_trace_path = 'output/' + task + '/' + mname + '.vali_acc.txt'
        vali_acc_trace_file = open(vali_acc_trace_path, 'w')
        vali_acc_trace_file.write('epoch batch vali_acc_cost\n')

    if SAVE_INTERNAL_GRADS:
        hidden_gradients_path = 'output/' + task + '/' + mname + '.hidden_gradients.txt'
        hidden_gradients_file = open(hidden_gradients_path, 'w')
        hidden_gradients_file.write('epoch batch k norm\n')
        hidden_states_path = 'output/' + task + '/' + mname + '.hidden_states.txt'
        hidden_states_file = open(hidden_states_path, 'w')
        hidden_states_file.write('epoch batch k value what\n')

    if TIMING:
        timing_path = 'output/' + task + '/' + mname + '.timing.txt'
        timing_file = open(timing_path, 'w')
        timing_file.write('epoch batch time\n')

    # === ops for training === #
    if verbose: print 'setting up train ops...'
    cost = get_cost(outputs, y, loss_type)

    if model in {'ortho_tanhRNN', 'uRNN'}:
        # COMMENCE GRADIENT HACKS
        opt = create_optimiser(learning_rate)
        nonU_variables = []
        if model == 'uRNN':
            U_re_name = 'RNN/uRNN/Unitary/Transition/Matrix/Real:0'
            U_im_name = 'RNN/uRNN/Unitary/Transition/Matrix/Imaginary:0'
            for var in tf.trainable_variables():
                print var.name
                if var.name == U_re_name:
                    U_re_variable = var
                elif var.name == U_im_name:
                    U_im_variable = var
                else:
                    nonU_variables.append(var)
            U_variables = [U_re_variable, U_im_variable]
            # WARNING: dtype
            U_new_re = tf.placeholder(dtype=tf.float32, shape=[state_size, state_size])
            U_new_im = tf.placeholder(dtype=tf.float32, shape=[state_size, state_size])
            # ops
            assign_re_op = assign_variable(U_re_variable, U_new_re)
            assign_im_op = assign_variable(U_im_variable, U_new_im)
        else:
            lambdas = np.random.normal(size=(state_size*(state_size-1)/2))
            # TODO: check this name
            U_name = 'RNN/tanhRNN/Linear/Transition/Matrix:0'
            for var in tf.trainable_variables():
                if var.name == U_name:
                    U_variable = var
                else:
                    nonU_variables.append(var)
            U_variables = [U_variable]
            U_new = tf.placeholder(dtype=tf.float32, shape=[state_size, state_size])
            # ops
            assign_op = assign_variable(U_variable, U_new)
        # get gradients (alternately: just store indices and separate afterwards)
        g_and_v_nonU = get_gradients(opt, cost, gradient_clipping, nonU_variables)
        g_and_v_U = get_gradients(opt, cost, gradient_clipping, U_variables)
        # normal train op
        train_op = update_variables(opt, g_and_v_nonU)
                    
        # save-specific thing: saving lambdas
        lambda_file = open('output/' + task + '/' + mname + '_lambdas.txt', 'w')
        lambda_file.write('batch ' + ' '.join(map(lambda x: 'lambda_' + str(x), xrange(len(lambdas)))) + '\n')
    else:
        train_op = update_step(cost, learning_rate, gradient_clipping)
   
    # === for checkpointing the model === #
    saver = tf.train.Saver()        # for checkpointing the model

    # === gpu stuff === #
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.9

    # === let's do it! === #
    if verbose: print 'initialising session...'
    with tf.Session(config=config) as session:
        # summaries
#        merged = tf.merge_all_summaries()
#        train_writer = tf.train.SummaryWriter('./log/' + model, session.graph)
        
        if verbose: print 'initialising variables...'
        session.run(tf.initialize_all_variables())

        # === get relevant operations for calculating internal gradient norms === #
        if SAVE_INTERNAL_GRADS:
            graph_ops = session.graph.get_operations()
            internal_grads = [None]*train_data.sequence_length
            internal_states = [None]*train_data.sequence_length
            o_counter = 0
            for o in graph_ops:
                if 'new_state' in o.name and not 'grad' in o.name:
                    # internal state
                    internal_grads[o_counter] = tf.gradients(cost, o.values()[0])[0]
                    internal_states[o_counter] = o.values()[0]
                    o_counter += 1
            if not o_counter == train_data.sequence_length:
                print o_counter, train_data.sequence_length

        # === train loop === #
        if verbose: print 'preparing to train!'
        for epoch in xrange(num_epochs):
            # shuffle the data at each epoch
            if verbose: print 'shuffling training data at epoch', epoch
            train_data.shuffle()
            for batch_index in xrange(num_batches):
                # definitely scope for fancy iterator but yolo
                batch_x, batch_y = train_data.get_batch(batch_index, batch_size)
         
                # === gradient hacks etc. === #
                # TODO: speed-profiling
                if model == 'uRNN' or model == 'ortho_tanhRNN':
                    if verbose: print 'preparing for gradient hacks'
                    # we can use the eigtrick, lambdas is defined...
                    if model == 'uRNN':
                        # extract dcost/dU terms from tf
                        dcost_dU_re, dcost_dU_im = session.run([g_and_v_U[0][0], g_and_v_U[1][0]], {x:batch_x, y:batch_y})
                        # calculate gradients of lambdas using eigenvalue decomposition trick
                        U_new_re_array, U_new_im_array, dlambdas = eigtrick_lambda_update(dcost_dU_re, dcost_dU_im, lambdas, learning_rate, speedy=True)
                        #train_cost = session.run(cost, {x:batch_x, y:batch_y})
                        _, _, _ = session.run([train_op, assign_re_op, assign_im_op], {x: batch_x, y:batch_y, U_new_re: U_new_re_array, U_new_im: U_new_im_array})
                    else:
                        #model == 'ortho_tanhRNN':
                        # extract dcost/dU terms from tf
                        dcost_dU_re = session.run(g_and_v_U[0][0], {x:batch_x, y:batch_y})
                        dcost_dU_im = np.zeros_like(dcost_dU_re)
                        # calculate gradients of lambdas using eigenvalue decomposition trick
                        U_new_re_array, U_new_im_array, dlambdas = eigtrick_lambda_update(dcost_dU_re, dcost_dU_im, lambdas, learning_rate, speedy=True)
                        assert np.array_equal(U_new_im_array, np.zeros_like(U_new_im_array))
                        # calculate train cost, update variables
#                        train_cost, _, _, summary = session.run([cost, train_op, assign_op, merged], {x: batch_x, y:batch_y, U_new: U_new_re_array})
                        train_cost, _, _ = session.run([cost, train_op, assign_op], {x: batch_x, y:batch_y, U_new: U_new_re_array})
               
                else:
                    if verbose: print 'calculating cost and updating parameters...'
                    # no eigtrick required, no numerical gradients, all is fine
                    train_cost, _ = session.run([cost, train_op], {x: batch_x, y: batch_y})
                
                # TODO OFF FOR NOW
#                train_writer.add_summary(summary, batch_index)

                if batch_index % 150 == 0:
                    train_cost = session.run(cost, {x:batch_x, y:batch_y})
                    print epoch, '\t', batch_index, '\t', loss_type + ':', train_cost
                    vali_cost = session.run(cost, {x: vali_data.x, y: vali_data.y})

                    train_trace_file.write(str(epoch) + ' ' + str(batch_index) + ' ' + str(train_cost) + '\n')
                    vali_trace_file.write(str(epoch) + ' ' + str(batch_index) + ' ' + str(vali_cost) + '\n')
                    train_trace_file.flush()
                    vali_trace_file.flush()
              
                    # save best parameters
                    if vali_cost < best_vali_cost:
                        best_vali_cost = vali_cost
                        save_path = saver.save(session, best_model_path)
                        print epoch, '\t', batch_index, '\t*** VALI', loss_type + ':', vali_cost, '\t('+save_path+')'
                    else:
                        print epoch, '\t', batch_index, '\t    VALI', loss_type + ':', vali_cost

                    # get mnist accuracy
                    if 'mnist' in task:
                        last_outs = session.run(outputs[-1], {x: vali_data.x, y:vali_data.y})
                        class_predictions = np.argmax(np.exp(last_outs)/np.sum(np.exp(last_outs), axis=1).reshape(6000, -1), axis=1)
                        vali_acc = 100 * np.mean(class_predictions == vali_data.y)
                        vali_acc_trace_file.write(str(epoch) + ' ' + str(batch_index) + ' ' + str(vali_acc) + '\n')
                        vali_acc_trace_file.flush()
                        print epoch, '\t', batch_index, '\t    VALI ACC:', vali_acc


                    # timing
                    if TIMING:
                        if batch_index == 0:
                            dt = 0.0
                            t_prev = time()
                        t = time()
                        dt = t - t_prev
                        print 'dt:', dt
                        timing_file.write(str(epoch) + ' ' + str(batch_index) + ' ' + str(dt) +'\n')
                        timing_file.flush()
                        t_prev = t

                if batch_index % 500 == 0 and model == 'uRNN':
                    lambda_file.write(str(batch_index) + ' ' + ' '.join(map(str, lambdas)) + '\n')

                # calculate gradients of cost with respect to internal states
                # save the mean (over the batch) norms of these
                if SAVE_INTERNAL_GRADS and batch_index % 500 == 0:
                    print 'calculating internal gradients...'
                    internal_grads_np = session.run(internal_grads, {x:batch_x, y:batch_y})
                    print 'calculating internal states...'
                    internal_states_np = session.run(internal_states, {x:batch_x, y:batch_y})
                    # get norms of each gradient vector, then average over the batch
                    for (k, grad_at_k) in enumerate(internal_grads_np):
                        norm_at_k = np.mean(np.linalg.norm(grad_at_k, axis=1))
                        hidden_gradients_file.write(str(epoch) + ' ' + str(batch_index) + ' ' + str(k) + ' ' + str(norm_at_k) + '\n')
                    # this is actually batch_size final states...
                    final_state = internal_states_np[-1]
                    for (k, state_batch) in enumerate(internal_states_np):
                        diff  = final_state - state_batch
                        mean_norm = np.mean(np.linalg.norm(state_batch, axis=1))
                        # get the norm of the difference, then average over the batch
                        mean_diff_norm = np.mean(np.linalg.norm(diff, axis=1))
                        hidden_states_file.write(str(epoch) + ' ' + str(batch_index) + ' ' + str(k) + ' ' + str(mean_diff_norm) + ' diff\n')
                        hidden_states_file.write(str(epoch) + ' ' + str(batch_index) + ' ' + str(k) + ' ' + str(mean_norm) + ' norm\n')
                    hidden_gradients_file.flush()
                    hidden_states_file.flush()
#                    if batch_index == 500:
#                        sys.exit('finished recording hidden state info')

        print 'Training completed.'
        if DO_TEST:
            print 'Loading best model from', best_model_path
            saver.restore(session, best_model_path)
            test_cost = session.run(cost, {x: test_data.x, y: test_data.y})
            print 'Performance on test set:', test_cost

# === parse inputs === #
parser = argparse.ArgumentParser(description='Run task of long-term memory capacity of RNN.')
parser.add_argument('--task', type=str, help='which task? adding/memory', 
                    default='adding')
parser.add_argument('--batch_size', type=int, 
                    default=20)
parser.add_argument('--state_size', type=int, help='size of internal state', 
                    default=5)
parser.add_argument('--T', type=int, help='memory time-scale or addition input length', 
                    default=100)
parser.add_argument('--model', type=str, help='which RNN model to use?', 
                    default='uRNN')
parser.add_argument('--data_path', type=str, help='path to dict of ExperimentData objects (if empty, generate data)', 
                    default='')
parser.add_argument('--learning_rate', type=float, help='prefactor of gradient in gradient descent parameter update', 
                    default=0.001)
parser.add_argument('--num_epochs', type=int, help='number of times to run through training data', 
                    default=10)
parser.add_argument('--identifier', type=str, help='a string to identify the experiment',
                    default='')
parser.add_argument('--verbose', type=bool, help='verbosity?',
                    default=False)
options = vars(parser.parse_args())

# === derivative options === #
if options['model'] in {'complex_RNN', 'ortho_tanhRNN', 'uRNN'}:
    options['gradient_clipping'] = False
else:
    options['gradient_clipping'] = True
    # turning off gradient clipping... 
#    options['gradient_clipping'] = False

# --- load pre-calculated data --- #
T = options['T']
if options['task'] == 'adding':
    if T == 100:
        options['data_path'] = 'input/adding/100000_10000_1479996206_100.pk'
    elif T == 200:
        options['data_path'] = ''
    elif T == 400:
        options['data_path'] = ''
    elif T == 750:
        options['data_path'] = ''
    else:
        options['data_path'] = ''
elif options['task'] == 'memory':
    if T == 100:
        options['data_path'] = 'input/memory/100000_10000_1479996618_100.pk'
    elif T == 200:
        options['data_path'] = ''
    elif T == 300:
        options['data_path'] = ''
    elif T == 500:
        options['data_path'] = ''
    else:
        options['data_path'] = ''
elif options['task'] == 'mnist':
    options['data_path'] = 'input/mnist/mnist.pk'    # (T is meaningless here...)
elif options['task'] == 'mnist_perm':
    options['data_path'] = 'input/mnist_perm/mnist_perm.pk'
else:
    raise ValueError(options['task'])

# === print stuff ===#
print 'Created dictionary of options'
for (key, value) in options.iteritems():
    print key, ':\t', value

# === now run (auto mode) === #
AUTO = True
if AUTO:
    save_options(options)
    run_experiment(**options)
