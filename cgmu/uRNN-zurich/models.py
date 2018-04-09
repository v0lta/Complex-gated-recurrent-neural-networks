# -*- coding: utf-8 -*-
import theano, cPickle
import theano.tensor as T
import numpy as np
import pdb
from fftconv import cufft, cuifft

def initialize_matrix(n_in, n_out, name, rng):
    bin = np.sqrt(6. / (n_in + n_out))
    # STEPH: again with the built-in names, sadface
    values = np.asarray(rng.uniform(low=-bin,
                                    high=bin,
                                    size=(n_in, n_out)),
                                    dtype=theano.config.floatX)
    return theano.shared(value=values, name=name)

def do_fft(input, n_hidden):
    fft_input = T.reshape(input, (input.shape[0], 2, n_hidden))
    fft_input = fft_input.dimshuffle(0,2,1)
    fft_output = cufft(fft_input) / T.sqrt(n_hidden)
    # STEPH: see fftconv for cufft
    fft_output = fft_output.dimshuffle(0,2,1)
    output = T.reshape(fft_output, (input.shape[0], 2*n_hidden))
    return output

def do_ifft(input, n_hidden):
    ifft_input = T.reshape(input, (input.shape[0], 2, n_hidden))
    ifft_input = ifft_input.dimshuffle(0,2,1)
    ifft_output = cuifft(ifft_input) / T.sqrt(n_hidden)
    ifft_output = ifft_output.dimshuffle(0,2,1)
    output = T.reshape(ifft_output, (input.shape[0], 2*n_hidden))
    return output


def times_diag(input, n_hidden, diag, swap_re_im):
    # STEPH: built-in names, sadface
    d = T.concatenate([diag, -diag])
    # STEPH: theano concatenation, minus for reasons yet unknown... something
    #   possibly to do with phase?
    #   TODO: resolve this uncertainty
    #   suspicion: to do with a + ib -> a - ib somehow, since sin(-x) = -sin(x)
    
    Re = T.cos(d).dimshuffle('x',0)
    Im = T.sin(d).dimshuffle('x',0)
    # STEPH: d is the phase of the complex number, splitting it into real
    #   and imaginary parts here

    input_times_Re = input * Re
    input_times_Im = input * Im
    # STEPH: np does row-wise multiplication between matrix and vectors

    output = input_times_Re + input_times_Im[:, swap_re_im]
   
    return output
    
    
def vec_permutation(input, index_permute):
    return input[:, index_permute]      

   
def times_reflection(input, n_hidden, reflection):
    # comments here are Steph working through the maths
    # OK so the equation they give is:
    #   (I - 2 outer(v, v*)/|v|**2) h
    # (v is the reflection, h is the input)
    # this gets us to: (using Einstein notation)
    #   h_i - (2/|v|**2) v_i v*_j h_j
    # Looking at the final few lines of this function, what we would like to
    # show is: (since :n_hidden is the imaginary part of the output tensor)
    #       re(v_i v*_j h_j) = d - c
    #       im(v_i v*_j h_j) = a + b
    #
    # v = mu + i nu
    # h = alpha + i beta
    # v_i v*_j h_j = (mu_i + i nu_i) (mu_j - i nu_j) (alpha_j + i beta_j)
    #       = (mu_i mu_j - i mu_i nu_j + i nu_i mu_j + nu_i nu_j) (alpha_j + i beta_j)
    #       = (mu_i mu_j alpha_j + i mu_i mu_j beta_j +
    #          -i mu_i nu_j alpha_j + mu_i nu_j beta_j +
    #           i nu_i mu_j alpha_j - nu_i mu_j beta_j +
    #             nu_i nu_j alpha_j + i nu_i nu_j beta_j) = K
    #
    # What an expression!
    # Let's split it up:
    # re(K) = (mu_i mu_j alpha_j + mu_i nu_j beta_j +
    #          -nu_i mu_j beta_j + nu_i nu_j alpha_j)
    # im(K) = (mu_i mu_j beta_j - mu_i nu_j alpha_j +
    #          + nu_i mu_j alpha_j + nu_i nu_j beta_j)
    #
    # Now let's replace the scalar parts (the repeated js...)
    # αμ = alpha_j mu_j
    # αν = alpha_j nu_j
    # βμ = beta_j mu_j
    # βν = beta_j nu_j
    #
    # re(K) = (mu_i αμ + mu_i βν - nu_i βμ + nu_i αν )
    # im(K) = (mu_i βμ - mu_i αν + nu_i αμ + nu_i βν )
    #
    # Simplifying further...
    #
    # re(K) = mu_i ( αμ + βν ) - nu_i ( βμ - αν ) = nope - nope
    # im(K) = mu_i ( βμ - αν ) + nu_i ( αμ + βν ) = nope + nope
    #
    # Jumping ahead (see below) to the definitions of a, b, c, d...
    #
    # a = mu_i ( αμ - βν )
    # b = nu_i ( αν + βμ )
    # c = nu_i ( αμ - βν )
    # d = mu_i ( αν + βμ )
    #
    # And so:
    # d - c = mu_i ( αν + βμ ) - nu_i ( αμ - βν )
    # a + b = mu_i ( αμ - βν ) + nu_i ( αν + βμ )
    #
    # ... huh, what is going on?
    # ... double-checking my maths!
    # ... double-checking their maths!
    # ... looks OK?
    # ... will need to TRIPLE-check my maths when it's not 1am.
    #
    # Possibility: when they used a * in the paper, they meant *transpose*
    # and not *conjugate transpose*...
    #
    # This would result in...
    #
    # v_i v_j h_j = (mu_i + i nu_i) (mu_j + i nu_j) (alpha_j + i beta_j)
    #       = (mu_i mu_j + i mu_i nu_j + i nu_i mu_j - nu_i nu_j) (alpha_j + i beta_j)
    #       = (mu_i mu_j alpha_j + i mu_i mu_j beta_j +
    #          + i mu_i nu_j alpha_j - mu_i nu_j beta_j +
    #           i nu_i mu_j alpha_j - nu_i mu_j beta_j +
    #           - nu_i nu_j alpha_j - i nu_i nu_j beta_j) = J
    #
    # re(J) = (mu_i mu_j alpha_j - mu_i nu_j beta_j +
    #          - nu_i mu_j beta_j - nu_i nu_j alpha_j)
    # im(J) = (mu_i mu_j beta_j + mu_i nu_j alpha_j +
    #            nu_i mu_j alpha_j - nu_i nu_j beta_j)
    #
    # Replacing scalar parts...
    # re(J) = mu_i αμ - mu_i βν - nu_i βμ - nu_i αν
    # im(J) = mu_i βμ + mu_i αν + nu_i αμ - nu_i βν
    #
    # Further simplifying...
    #
    # re(J) = mu_i ( αμ - βν ) - nu_i ( βμ + αν ) = a - b
    # im(J) = mu_i ( βμ + αν ) + nu_i ( αμ - βν ) = d + c
    #
    # ... closer but NOT THE SAME
    # WHAT IS GOING ON HERE?

    input_re = input[:, :n_hidden]
    # alpha
    input_im = input[:, n_hidden:]
    # beta
    reflect_re = reflection[:n_hidden]
    # mu
    reflect_im = reflection[n_hidden:]
    # nu

    vstarv = (reflection**2).sum()

    # (the following things are roughly scalars)
    # (they actually are as long as the batch size, e.g. input[0])
    input_re_reflect_re = T.dot(input_re, reflect_re)
    # αμ
    input_re_reflect_im = T.dot(input_re, reflect_im)
    # αν
    input_im_reflect_re = T.dot(input_im, reflect_re)
    # βμ
    input_im_reflect_im = T.dot(input_im, reflect_im)
    # βν

#
    a = T.outer(input_re_reflect_re - input_im_reflect_im, reflect_re)
    # outer(αμ - βν, mu)
    b = T.outer(input_re_reflect_im + input_im_reflect_re, reflect_im)
    # outer(αν + βμ, nu)
    c = T.outer(input_re_reflect_re - input_im_reflect_im, reflect_im)
    # outer(αμ - βν, nu)
    d = T.outer(input_re_reflect_im + input_im_reflect_re, reflect_re)
    # outer(αν + βμ, mu)

    output = input
    output = T.inc_subtensor(output[:, :n_hidden], - 2. / vstarv * (a + b))
    output = T.inc_subtensor(output[:, n_hidden:], - 2. / vstarv * (d - c))

    return output
#
def compute_cost_t(lin_output, loss_function, y_t):
    if loss_function == 'CE':
        RNN_output = T.nnet.softmax(lin_output)
        cost_t = T.nnet.categorical_crossentropy(RNN_output, y_t).mean()
        # TODO: (STEPH) review maths on this
        acc_t =(T.eq(T.argmax(RNN_output, axis=-1), y_t)).mean(dtype=theano.config.floatX)
    elif loss_function == 'MSE':
        cost_t = ((lin_output - y_t)**2).mean()
        acc_t = theano.shared(np.float32(0.0))

    return cost_t, acc_t


def initialize_data_nodes(loss_function, input_type, out_every_t):
    # initialises the theano objects for data and labels
    x = T.tensor3() if input_type == 'real' else T.matrix(dtype='int32')
    # STEPH: x is either real or ... integers?
    if loss_function == 'CE':
        y = T.matrix(dtype='int32') if out_every_t else T.vector(dtype='int32')
    else:  
        # STEPH: if not CE, then RSE, btw...
        y = T.tensor3() if out_every_t else T.matrix()
    return x, y        



def IRNN(n_input, n_hidden, n_output, input_type='real', out_every_t=False, loss_function='CE'):
    # STEPH: this differs from tanhRNN in two places, see below
    np.random.seed(1234)
    rng = np.random.RandomState(1234)

    x, y = initialize_data_nodes(loss_function, input_type, out_every_t)
    inputs = [x, y]

    h_0 = theano.shared(np.zeros((1, n_hidden), dtype=theano.config.floatX))
    V = initialize_matrix(n_input, n_hidden, 'V', rng)
    W = theano.shared(np.identity(n_hidden, dtype=theano.config.floatX))
    # STEPH: W differs from that of tanhRNN: this is just identity!
    out_mat = initialize_matrix(n_hidden, n_output, 'out_mat', rng)
    hidden_bias = theano.shared(np.zeros((n_hidden,), dtype=theano.config.floatX))
    out_bias = theano.shared(np.zeros((n_output,), dtype=theano.config.floatX))

    parameters = [h_0, V, W, out_mat, hidden_bias, out_bias]

    def recurrence(x_t, y_t, h_prev, cost_prev, acc_prev, V, W, hidden_bias, out_mat, out_bias):
        if loss_function == 'CE':
            data_lin_output = V[x_t]
        else:
            data_lin_output = T.dot(x_t, V)
        
        h_t = T.nnet.relu(T.dot(h_prev, W) + data_lin_output + hidden_bias.dimshuffle('x', 0))
        # STEPH: differs from tanhRNN: here we have relu, there they had tanh
        if out_every_t:
            lin_output = T.dot(h_t, out_mat) + out_bias.dimshuffle('x', 0)
            cost_t, acc_t = compute_cost_t(lin_output, loss_function, y_t)
        else:
            cost_t = theano.shared(np.float32(0.0))
            acc_t = theano.shared(np.float32(0.0))
 
        return h_t, cost_t, acc_t
    
    non_sequences = [V, W, hidden_bias, out_mat, out_bias]

    h_0_batch = T.tile(h_0, [x.shape[1], 1])

    if out_every_t:
        sequences = [x, y]
    else:
        sequences = [x, T.tile(theano.shared(np.zeros((1,1), dtype=theano.config.floatX)), [x.shape[0], 1, 1])]

    outputs_info = [h_0_batch, theano.shared(np.float32(0.0)), theano.shared(np.float32(0.0))]
    
    [hidden_states, cost_steps, acc_steps], updates = theano.scan(fn=recurrence,
                                                                  sequences=sequences,
                                                                  non_sequences=non_sequences,
                                                                  outputs_info = outputs_info)
   
    if not out_every_t:
        lin_output = T.dot(hidden_states[-1,:,:], out_mat) + out_bias.dimshuffle('x', 0)
        costs = compute_cost_t(lin_output, loss_function, y)
    else:
        cost = cost_steps.mean()
        accuracy = acc_steps.mean()
        costs = [cost, accuracy]

    return inputs, parameters, costs



def tanhRNN(n_input, n_hidden, n_output, input_type='real', out_every_t=False, loss_function='CE'):
    np.random.seed(1234)
    rng = np.random.RandomState(1234)
    # STEPH: initialising np's generic RNG and a specific rng identically
    #   uncertain why but maybe we'll find out soon

    x, y = initialize_data_nodes(loss_function, input_type, out_every_t)
    inputs = [x, y]

    h_0 = theano.shared(np.zeros((1, n_hidden), dtype=theano.config.floatX))
    V = initialize_matrix(n_input, n_hidden, 'V', rng)
    W = initialize_matrix(n_hidden, n_hidden, 'W', rng)
    # STEPH: W is the weights of the recurrence (can tell cause of its shape!)
    out_mat = initialize_matrix(n_hidden, n_output, 'out_mat', rng)
    hidden_bias = theano.shared(np.zeros((n_hidden,), dtype=theano.config.floatX))
    out_bias = theano.shared(np.zeros((n_output,), dtype=theano.config.floatX))
    parameters = [h_0, V, W, out_mat, hidden_bias, out_bias]

    def recurrence(x_t, y_t, h_prev, cost_prev, acc_prev, V, W, hidden_bias, out_mat, out_bias):
        # all of this is to get the hidden state, and possibly cost/accuracy
        if loss_function == 'CE':
            data_lin_output = V[x_t]
            # STEPH: uncertain why this is named thusly
            # STEPH: in CE case, the data is just an index, I guess...
            #   basically, an indicator vector
            #   I think this may be confounded with the experimental setup
            #   CE appears in ?
        else:
            data_lin_output = T.dot(x_t, V)
            # STEPH: 'as normal', folding the data from the sequence in
   
        h_t = T.tanh(T.dot(h_prev, W) + data_lin_output + hidden_bias.dimshuffle('x', 0))
        # STEPH: dimshuffle (theano) here, makes row out of 1d vector, N -> 1xN
        if out_every_t:
            lin_output = T.dot(h_t, out_mat) + out_bias.dimshuffle('x', 0)
            cost_t, acc_t = compute_cost_t(lin_output, loss_function, y_t)
        else:
            # STEPH: no cost/accuracy until the end!
            cost_t = theano.shared(np.float32(0.0))
            acc_t = theano.shared(np.float32(0.0))

        return h_t, cost_t, acc_t 
    
    non_sequences = [V, W, hidden_bias, out_mat, out_bias]
    # STEPH: naming due to scan (theano); these are 'fixed' values in scan

    h_0_batch = T.tile(h_0, [x.shape[1], 1])
    # STEPH: tile (theano) repeats input x according to pattern
    #   pattern is number of times to tile in each direction

    if out_every_t:
        sequences = [x, y]
    else:
        # STEPH: the 'y' here is just... a bunch of weirdly-shaped zeros?
        sequences = [x, T.tile(theano.shared(np.zeros((1,1), dtype=theano.config.floatX)), [x.shape[0], 1, 1])]
    # STEPH: sequences here are the input we loop over...

    outputs_info = [h_0_batch, theano.shared(np.float32(0.0)), theano.shared(np.float32(0.0))]
    # STEPH: naming due to scan, these are initialisation values... see return
    # value of recurrence: h_t, cost_t, acc_t...
    
    [hidden_states, cost_steps, acc_steps], updates = theano.scan(fn=recurrence,
                                                                  sequences=sequences,
                                                                  non_sequences=non_sequences,
                                                                  outputs_info=outputs_info)
    # STEPH: remembering how to do scan!
    #   outputs_info: contains initialisation, naming is bizarre, whatever
    #   non_sequences: unchanging variables
    #   sequences: tensors to be looped over
    #   so fn receives (sequences, previous output, non_sequences):
    #       this seems to square with the order of arguments in 'recurrence'
    #       TODO: read scan more carefully to confirm this
   
    if not out_every_t:
        lin_output = T.dot(hidden_states[-1,:,:], out_mat) + out_bias.dimshuffle('x', 0)
        costs = compute_cost_t(lin_output, loss_function, y)
        # STEPH: cost is computed off the final hidden state
    else:
        cost = cost_steps.mean()
        accuracy = acc_steps.mean()
        costs = [cost, accuracy]

    return inputs, parameters, costs



def LSTM(n_input, n_hidden, n_output, input_type='real', out_every_t=False, loss_function='CE'):
    np.random.seed(1234)
    rng = np.random.RandomState(1234)

    # STEPH: i for input, f for forget, c for candidate, o for output
    W_i = initialize_matrix(n_input, n_hidden, 'W_i', rng)
    W_f = initialize_matrix(n_input, n_hidden, 'W_f', rng)
    W_c = initialize_matrix(n_input, n_hidden, 'W_c', rng)
    W_o = initialize_matrix(n_input, n_hidden, 'W_o', rng)
    U_i = initialize_matrix(n_hidden, n_hidden, 'U_i', rng)
    U_f = initialize_matrix(n_hidden, n_hidden, 'U_f', rng)
    U_c = initialize_matrix(n_hidden, n_hidden, 'U_c', rng)
    U_o = initialize_matrix(n_hidden, n_hidden, 'U_o', rng)
    # STEPH: note that U is not out_mat as it was in complex_RNN
    V_o = initialize_matrix(n_hidden, n_hidden, 'V_o', rng)
    b_i = theano.shared(np.zeros((n_hidden,), dtype=theano.config.floatX))
    b_f = theano.shared(np.ones((n_hidden,), dtype=theano.config.floatX))
    b_c = theano.shared(np.zeros((n_hidden,), dtype=theano.config.floatX))
    b_o = theano.shared(np.zeros((n_hidden,), dtype=theano.config.floatX))
    h_0 = theano.shared(np.zeros((1, n_hidden), dtype=theano.config.floatX))
    state_0 = theano.shared(np.zeros((1, n_hidden), dtype=theano.config.floatX))
    out_mat = initialize_matrix(n_hidden, n_output, 'out_mat', rng)
    out_bias = theano.shared(np.zeros((n_output,), dtype=theano.config.floatX))
    parameters = [W_i, W_f, W_c, W_o, U_i, U_f, U_c, U_o, V_o, b_i, b_f, b_c, b_o, h_0, state_0, out_mat, out_bias]

    x, y = initialize_data_nodes(loss_function, input_type, out_every_t)
    
    def recurrence(x_t, y_t, h_prev, state_prev, cost_prev, acc_prev,
                   W_i, W_f, W_c, W_o, U_i, U_f, U_c, U_o, V_o, b_i, b_f, b_c, b_o, out_mat, out_bias):
        
        if loss_function == 'CE':
            x_t_W_i = W_i[x_t]
            x_t_W_c = W_c[x_t]
            x_t_W_f = W_f[x_t]
            x_t_W_o = W_o[x_t]
        else:
            x_t_W_i = T.dot(x_t, W_i)
            x_t_W_c = T.dot(x_t, W_c)
            x_t_W_f = T.dot(x_t, W_f)
            x_t_W_o = T.dot(x_t, W_o)
            
        input_t = T.nnet.sigmoid(x_t_W_i + T.dot(h_prev, U_i) + b_i.dimshuffle('x', 0))
        # STEPH: save candidate?
        candidate_t = T.tanh(x_t_W_c + T.dot(h_prev, U_c) + b_c.dimshuffle('x', 0))
        forget_t = T.nnet.sigmoid(x_t_W_f + T.dot(h_prev, U_f) + b_f.dimshuffle('x', 0))
        # STEPH: forget previosu state?

        state_t = input_t * candidate_t + forget_t * state_prev
        # STEPH: so we can both save the input and not forget the previous, OK

        output_t = T.nnet.sigmoid(x_t_W_o + T.dot(h_prev, U_o) + T.dot(state_t, V_o) + b_o.dimshuffle('x', 0))
        # TODO: (STEPH) double-check maths, here!

        h_t = output_t * T.tanh(state_t)

        # STEPH: same  as other models...
        if out_every_t:
            lin_output = T.dot(h_t, out_mat) + out_bias.dimshuffle('x', 0)
            cost_t, acc_t = compute_cost_t(lin_output, loss_function, y_t)
        else:
            cost_t = theano.shared(np.float32(0.0))
            acc_t = theano.shared(np.float32(0.0))
 
        return h_t, state_t, cost_t, acc_t

    non_sequences = [W_i, W_f, W_c, W_o, U_i, U_f, U_c, U_o, V_o, b_i, b_f, b_c, b_o, out_mat, out_bias]
    
    # STEPH: same as tanhRNN, etc... the scan part is generally duplicated!
    h_0_batch = T.tile(h_0, [x.shape[1], 1])
    state_0_batch = T.tile(state_0, [x.shape[1], 1])
    
    if out_every_t:
        sequences = [x, y]
    else:
        sequences = [x, T.tile(theano.shared(np.zeros((1,1), dtype=theano.config.floatX)), [x.shape[0], 1, 1])]

    outputs_info = [h_0_batch, state_0_batch, theano.shared(np.float32(0.0)), theano.shared(np.float32(0.0))]
        
    [hidden_states, states, cost_steps, acc_steps], updates = theano.scan(fn=recurrence,
                                                                          sequences=sequences,
                                                                          non_sequences=non_sequences,
                                                                          outputs_info=outputs_info)
    
    if not out_every_t:
        lin_output = T.dot(hidden_states[-1,:,:], out_mat) + out_bias.dimshuffle('x', 0)
        costs = compute_cost_t(lin_output, loss_function, y)            
    else:
        cost = cost_steps.mean()
        accuracy = acc_steps.mean()
        costs = [cost, accuracy]

    return [x, y], parameters, costs


def complex_RNN(n_input, n_hidden, n_output, input_type='real', out_every_t=False, loss_function='CE'):

    np.random.seed(1234)
    rng = np.random.RandomState(1234)

    # Initialize parameters: theta, V_re, V_im, hidden_bias, U, out_bias, h_0
    V = initialize_matrix(n_input, 2*n_hidden, 'V', rng)
    U = initialize_matrix(2 * n_hidden, n_output, 'U', rng)
    # STEPH: U was previously known as out_mat
    hidden_bias = theano.shared(np.asarray(rng.uniform(low=-0.01,
                                                       high=0.01,
                                                       size=(n_hidden,)),
                                           dtype=theano.config.floatX), 
                                name='hidden_bias')
    # STEPH: hidden bias is simply initialised differently in this case 
    reflection = initialize_matrix(2, 2*n_hidden, 'reflection', rng)
    # STEPH: part of recurrence (~W)
    out_bias = theano.shared(np.zeros((n_output,), dtype=theano.config.floatX), name='out_bias')
    theta = theano.shared(np.asarray(rng.uniform(low=-np.pi,
                                                 high=np.pi,
                                                 size=(3, n_hidden)),
                                     dtype=theano.config.floatX), 
                                name='theta')
    # STEPH: theta is used in recurrence several times (~W)
    bucket = np.sqrt(3. / 2 / n_hidden) 
    h_0 = theano.shared(np.asarray(rng.uniform(low=-bucket,
                                               high=bucket,
                                               size=(1, 2 * n_hidden)), 
                                   dtype=theano.config.floatX),
                        name='h_0')
    # STEPH: special way of initialising hidden state
    parameters = [V, U, hidden_bias, reflection, out_bias, theta, h_0]

    x, y = initialize_data_nodes(loss_function, input_type, out_every_t)
    
    index_permute = np.random.permutation(n_hidden)
    # STEPH: permutation used in recurrence (~W)

    index_permute_long = np.concatenate((index_permute, index_permute + n_hidden))
    # STEPH: do the same permutation to both real and imaginary parts
    swap_re_im = np.concatenate((np.arange(n_hidden, 2*n_hidden), np.arange(n_hidden)))
    # STEPH: this is a permutation which swaps imaginary and real indices
    
    # define the recurrence used by theano.scan
    def recurrence(x_t, y_t, h_prev, cost_prev, acc_prev, theta, V, hidden_bias, out_bias, U):  

        # Compute hidden linear transform
        # STEPH: specific set of transformations, sliiightly not that important
        step1 = times_diag(h_prev, n_hidden, theta[0,:], swap_re_im)
        step2 = do_fft(step1, n_hidden)
        step3 = times_reflection(step2, n_hidden, reflection[0,:])
        step4 = vec_permutation(step3, index_permute_long)
        step5 = times_diag(step4, n_hidden, theta[1,:], swap_re_im)
        step6 = do_ifft(step5, n_hidden)
        step7 = times_reflection(step6, n_hidden, reflection[1,:])
        step8 = times_diag(step7, n_hidden, theta[2,:], swap_re_im)     
        
        hidden_lin_output = step8
        # STEPH: hidden_lin_output isn't complex enough to have its own name
        #   in the other models
        
        # Compute data linear transform
        if loss_function == 'CE':
            data_lin_output = V[T.cast(x_t, 'int32')]
        else:
            data_lin_output = T.dot(x_t, V)
            
        # Total linear output        
        lin_output = hidden_lin_output + data_lin_output


        # Apply non-linearity ----------------------------

        # scale RELU nonlinearity
        modulus = T.sqrt(lin_output**2 + lin_output[:, swap_re_im]**2)
        # STEPH: I think this comes to twice the modulus...
        #   TODO: check that
        rescale = T.maximum(modulus + T.tile(hidden_bias, [2]).dimshuffle('x', 0), 0.) / (modulus + 1e-5)
        h_t = lin_output * rescale
        
        if out_every_t:
            lin_output = T.dot(h_t, U) + out_bias.dimshuffle('x', 0)
            cost_t, acc_t = compute_cost_t(lin_output, loss_function, y_t)
        else:
            cost_t = theano.shared(np.float32(0.0))
            acc_t = theano.shared(np.float32(0.0))
        
        return h_t, cost_t, acc_t

    # compute hidden states
    # STEPH: the same as in tanhRNN, here (except U ~ out_mat)
    h_0_batch = T.tile(h_0, [x.shape[1], 1])
    non_sequences = [theta, V, hidden_bias, out_bias, U]
    if out_every_t:
        sequences = [x, y]
    else:
        sequences = [x, T.tile(theano.shared(np.zeros((1,1), dtype=theano.config.floatX)), [x.shape[0], 1, 1])]

    outputs_info=[h_0_batch, theano.shared(np.float32(0.0)), theano.shared(np.float32(0.0))]
    
    [hidden_states, cost_steps, acc_steps], updates = theano.scan(fn=recurrence,
                                                                  sequences=sequences,
                                                                  non_sequences=non_sequences,
                                                                  outputs_info=outputs_info)

    if not out_every_t:
        lin_output = T.dot(hidden_states[-1,:,:], U) + out_bias.dimshuffle('x', 0)
        costs = compute_cost_t(lin_output, loss_function, y)
    else:
        cost = cost_steps.mean()
        accuracy = acc_steps.mean()
        costs = [cost, accuracy]

    return [x, y], parameters, costs
    # STEPH: note that tanhRNN and IRNN return 'inputs' (= [x, y]), whereas
    #   complex_RNN and LSTM return [x, y]... I think this should not matter
    #   as x and y are (in theory) unchanged, but I'm still making a note of it.
    #

def orthogonal_RNN(n_input, n_hidden, n_output, input_type='real', out_every_t=False, loss_function='CE', basis=None):
    np.random.seed(1234)
    rng = np.random.RandomState(1234)
    
    x, y = initialize_data_nodes(loss_function, input_type, out_every_t)
    inputs = [x, y]

    # ---- encoder ---- #
    V = initialize_matrix(n_input, n_hidden, 'V', rng)
    # ---- decoder ---- #
    U = initialize_matrix(n_hidden, n_output, 'U', rng)
    out_bias = theano.shared(np.zeros((n_output,), dtype=theano.config.floatX), name='out_bias')
    # ---- hidden part ---- #
    dim_of_lie_algebra = n_hidden*(n_hidden-1)/2
    lambdas = theano.shared(np.asarray(rng.uniform(low=-1,
                                                   high=1,
                                                   size=(dim_of_lie_algebra,)),
                                       dtype=theano.config.floatX),
                            name='lambdas')
    # warning: symbolic_basis is expensive, memory-wise!
    if basis is None:
        symbolic_basis = theano.shared(np.asarray(rng.normal(size=(dim_of_lie_algebra,
                                                                   n_hidden,
                                                                   n_hidden)),
                                                  dtype=theano.config.floatX),
                                       name='symbolic_basis')
    else:
        symbolic_basis = theano.shared(basis, name='symbolic_basis')
    # here it is!
    #O = T.expm(T.dot(lambdas, symbolic_basis))
    # YOLO
    #O = T.tensordot(lambdas, symbolic_basis, axes=[0, 0])
    #O = lambdas[0]*symbolic_basis[0] + lambdas[10]*symbolic_basis[10]
    O = lambdas[dim_of_lie_algebra-1]*symbolic_basis[0]
    #lambdas[n_hidden*(n_hidden-1)/2 -1]*symbolic_basis[n_hidden*(n_hidden-1)/2 -1]
    # RIDICULOUS HACK THEANO IS WEIRD
    #for k in xrange(1, n_hidden*(n_hidden-1)/2):
#        O += lambdas[k]*symbolic_basis[k]
#    pdb.set_trace()
    #O = T.eye(n_hidden, n_hidden)
    # END YOLO
    # TODO: check maths on bucket
    bucket = np.sqrt(3. / 2 / n_hidden) 
    h_0 = theano.shared(np.asarray(rng.uniform(low=-bucket,
                                               high=bucket,
                                               size=(1, n_hidden)), 
                                   dtype=theano.config.floatX),
                        name='h_0')
    hidden_bias = theano.shared(np.asarray(rng.uniform(low=-0.01,
                                                       high=0.01,
                                                       size=(n_hidden,)),
                                           dtype=theano.config.floatX), 
                                name='hidden_bias')
   
    # ---- all the parameters! ---- #
    parameters = [V, U, out_bias, lambdas, h_0, hidden_bias]

    def recurrence(x_t, y_t, h_prev, cost_prev, acc_prev, V, O, hidden_bias, out_bias, U):  
        if loss_function == 'CE':
            # STEPH: why is this cast here???
            data_lin_output = V[T.cast(x_t, 'int32')]
        else:
            data_lin_output = T.dot(x_t, V)
        h_t = T.nnet.relu(T.dot(h_prev, O) + data_lin_output + hidden_bias.dimshuffle('x', 0))

        if out_every_t:
            lin_output = T.dot(h_t, U) + out_bias.dimshuffle('x', 0)
            cost_t, acc_t = compute_cost_t(lin_output, loss_function, y_t)
        else:
            cost_t = theano.shared(np.float32(0.0))
            acc_t = theano.shared(np.float32(0.0))
        
        return h_t, cost_t, acc_t

    # compute hidden states
    h_0_batch = T.tile(h_0, [x.shape[1], 1])
    non_sequences = [V, O, hidden_bias, out_bias, U]
    if out_every_t:
        sequences = [x, y]
    else:
        sequences = [x, T.tile(theano.shared(np.zeros((1,1), dtype=theano.config.floatX)), [x.shape[0], 1, 1])]

    outputs_info=[h_0_batch, theano.shared(np.float32(0.0)), theano.shared(np.float32(0.0))]
    
    [hidden_states, cost_steps, acc_steps], updates = theano.scan(fn=recurrence,
                                                                  sequences=sequences,
                                                                  non_sequences=non_sequences,
                                                                  outputs_info=outputs_info)

    if not out_every_t:
        lin_output = T.dot(hidden_states[-1,:,:], U) + out_bias.dimshuffle('x', 0)
        costs = compute_cost_t(lin_output, loss_function, y)
    else:
        cost = cost_steps.mean()
        accuracy = acc_steps.mean()
        costs = [cost, accuracy]

    return inputs, parameters, costs
 
def general_unitary_RNN(n_input, n_hidden, n_output, input_type='real', out_every_t=False, loss_function='CE'):
    # STEPH: hey, it's mine! copying proclivity towards boilerplate from rest
    #   of code: this is derived from complex_RNN!
    np.random.seed(1234)
    rng = np.random.RandomState(1234)

    # TODO: all from here (requires some engineering thoughts)
    # TODO TODO TODO
    # Initialize parameters: theta, V_re, V_im, hidden_bias, U, out_bias, h_0
    V = initialize_matrix(n_input, 2*n_hidden, 'V', rng)
    U = initialize_matrix(2 * n_hidden, n_output, 'U', rng)
    # STEPH: U was previously known as out_mat
    hidden_bias = theano.shared(np.asarray(rng.uniform(low=-0.01,
                                                       high=0.01,
                                                       size=(n_hidden,)),
                                           dtype=theano.config.floatX), 
                                name='hidden_bias')
    # STEPH: hidden bias is simply initialised differently in this case 
    reflection = initialize_matrix(2, 2*n_hidden, 'reflection', rng)
    # STEPH: part of recurrence (~W)
    out_bias = theano.shared(np.zeros((n_output,), dtype=theano.config.floatX), name='out_bias')
    theta = theano.shared(np.asarray(rng.uniform(low=-np.pi,
                                                 high=np.pi,
                                                 size=(3, n_hidden)),
                                     dtype=theano.config.floatX), 
                                name='theta')
    # STEPH: theta is used in recurrence several times (~W)
    bucket = np.sqrt(3. / 2 / n_hidden) 
    h_0 = theano.shared(np.asarray(rng.uniform(low=-bucket,
                                               high=bucket,
                                               size=(1, 2 * n_hidden)), 
                                   dtype=theano.config.floatX),
                        name='h_0')
    # STEPH: special way of initialising hidden state
    parameters = [V, U, hidden_bias, reflection, out_bias, theta, h_0]

    x, y = initialize_data_nodes(loss_function, input_type, out_every_t)
    
    index_permute = np.random.permutation(n_hidden)
    # STEPH: permutation used in recurrence (~W)

    index_permute_long = np.concatenate((index_permute, index_permute + n_hidden))
    # STEPH: do the same permutation to both real and imaginary parts
    swap_re_im = np.concatenate((np.arange(n_hidden, 2*n_hidden), np.arange(n_hidden)))
    # STEPH: this is a permutation which swaps imaginary and real indices
    
    # define the recurrence used by theano.scan
    def recurrence(x_t, y_t, h_prev, cost_prev, acc_prev, theta, V, hidden_bias, out_bias, U):  

        # Compute hidden linear transform
        # STEPH: specific set of transformations, sliiightly not that important
        step1 = times_diag(h_prev, n_hidden, theta[0,:], swap_re_im)
        step2 = do_fft(step1, n_hidden)
        step3 = times_reflection(step2, n_hidden, reflection[0,:])
        step4 = vec_permutation(step3, index_permute_long)
        step5 = times_diag(step4, n_hidden, theta[1,:], swap_re_im)
        step6 = do_ifft(step5, n_hidden)
        step7 = times_reflection(step6, n_hidden, reflection[1,:])
        step8 = times_diag(step7, n_hidden, theta[2,:], swap_re_im)     
        
        hidden_lin_output = step8
        # STEPH: hidden_lin_output isn't complex enough to have its own name
        #   in the other models
        
        # Compute data linear transform
        if loss_function == 'CE':
            data_lin_output = V[T.cast(x_t, 'int32')]
        else:
            data_lin_output = T.dot(x_t, V)
            
        # Total linear output        
        lin_output = hidden_lin_output + data_lin_output


        # Apply non-linearity ----------------------------

        # scale RELU nonlinearity
        modulus = T.sqrt(lin_output**2 + lin_output[:, swap_re_im]**2)
        # STEPH: I think this comes to twice the modulus...
        #   TODO: check that
        rescale = T.maximum(modulus + T.tile(hidden_bias, [2]).dimshuffle('x', 0), 0.) / (modulus + 1e-5)
        h_t = lin_output * rescale
        
        if out_every_t:
            lin_output = T.dot(h_t, U) + out_bias.dimshuffle('x', 0)
            cost_t, acc_t = compute_cost_t(lin_output, loss_function, y_t)
        else:
            cost_t = theano.shared(np.float32(0.0))
            acc_t = theano.shared(np.float32(0.0))
        
        return h_t, cost_t, acc_t

    # compute hidden states
    # STEPH: the same as in tanhRNN, here (except U ~ out_mat)
    h_0_batch = T.tile(h_0, [x.shape[1], 1])
    non_sequences = [theta, V, hidden_bias, out_bias, U]
    if out_every_t:
        sequences = [x, y]
    else:
        sequences = [x, T.tile(theano.shared(np.zeros((1,1), dtype=theano.config.floatX)), [x.shape[0], 1, 1])]

    outputs_info=[h_0_batch, theano.shared(np.float32(0.0)), theano.shared(np.float32(0.0))]
    
    [hidden_states, cost_steps, acc_steps], updates = theano.scan(fn=recurrence,
                                                                  sequences=sequences,
                                                                  non_sequences=non_sequences,
                                                                  outputs_info=outputs_info)

    if not out_every_t:
        lin_output = T.dot(hidden_states[-1,:,:], U) + out_bias.dimshuffle('x', 0)
        costs = compute_cost_t(lin_output, loss_function, y)
    else:
        cost = cost_steps.mean()
        accuracy = acc_steps.mean()
        costs = [cost, accuracy]

    return [x, y], parameters, costs
    # STEPH: note that tanhRNN and IRNN return 'inputs' (= [x, y]), whereas
    #   complex_RNN and LSTM return [x, y]... I think this should not matter
    #   as x and y are (in theory) unchanged, but I'm still making a note of it.
