#!/usr/bin/env ipython
# VISUALISATION IN PYTHON
# ... just cost curves etc.

import cPickle
import sys
#import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
plt.clf()

BATCH_SIZE = 20

# --- initialise --- #
# grab inputs
T = int(sys.argv[1])
task = sys.argv[2]
if task == 'adding':
    score = 'MSE'
elif task == 'memory':
    score = 'CE'
else:
    sys.exit('Unknown task', task)
plot_test = sys.argv[3] == 'test'
if plot_test:
    scaling_factor = 50*BATCH_SIZE
    plot_fname = task+'_'+str(T)+'_test.png'
else:
    scaling_factor = BATCH_SIZE
    plot_fname = task+'_'+str(T)+'_train.png'

print plot_fname

# constants etc.
#   we get train loss after every iteration
#   we get test loss after every 50 iterations
#   each iteration contains n_batch training examples
n_batch = 20
models = ['RNN', 'IRNN', 'LSTM', 'complex_RNN']
colours = {'RNN': 'r', 'IRNN': 'pink', 'LSTM': 'g', 'complex_RNN': 'b'}

# --- load all the data --- #
# (very deep dicts)
traces = dict()
for model in models:
    trace_file = task + '_' + model + '_' + str(T)
    try:
        traces[model] = cPickle.load(open(trace_file))
        # check consistency
        assert traces[model]['time_steps'] == T
        assert traces[model]['model'] == model
        print 'Loaded', trace_file
    except IOError:
        print 'Missing:', trace_file

# --- create plots --- #
# test or train?
if plot_test:
    loss = 'test_loss'
else:
    loss = 'train_loss'

# determine y/x min/max
xmax = 0
ymax = -1000
ymin = 1000
for (model, trace) in traces.iteritems():
    n_train = len(trace[loss])
    if n_train > xmax:
        xmax = n_train
    train_max = np.nanmax(trace[loss])
    if train_max > ymax:
        ymax = train_max
    train_min = np.nanmin(trace[loss])
    if train_min < ymin:
        ymin = train_min

xmax = scaling_factor*xmax
print 0, xmax
print ymin, ymax
# fix it anyway, ok
ymax = 1.0
ymin = -0.001

plt.axis([0, xmax, ymin, ymax])
# construct the arguments to plot
data_series = dict()
for model in traces.keys():
    series_x = scaling_factor*np.arange(len(traces[model][loss]))
    series_y = np.array(traces[model][loss])
    colour = colours[model]
    data_series[model], = plt.plot(series_x, series_y, colour, label=model, alpha=0.8)
plt.xlabel("training examples")
plt.ylabel(score)
plt.legend(loc='upper right')
plt.title(loss)
# now save ok
plt.savefig(plot_fname)
