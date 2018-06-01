from helper_module import *
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
from IPython.core.debugger import Tracer
debug_here = Tracer()

const_path = '/home/moritz/infcuda/complex_net_project/cgmu/phase-filter-RNN/'
bench_path = const_path + 'logs/paper_benchmarks/'

base_path = '/home/moritz/infcuda/complex_net_project/cgmu/phase-filter-RNN/logs/Loop_exp3/'
p1 = bench_path + '2018-05-08 21:50:18_adding_250_1000000_10000_128_0.001_50_LSTMCell_pt_2560'
p2 = base_path + '2018-05-04 05:25:32_adding_250_1000000_10000_128_0.001_50_UnitaryCell__activation_hirose__arjovski_basis_False_nat_grad_rms_False_qr_steps_None'
p3 = base_path + '2018-05-04 03:33:18_adding_250_1000000_10000_128_0.001_50_UnitaryCell__activation_mod_relu__arjovski_basis_False_nat_grad_rms_False_qr_steps_None'
p4 = base_path + ''
ps = [p1, p2, p3]
psf = []
for p in ps:
    for _, _, files in os.walk(p):
        psf.append(p + '/' + files[0])
window_size = 25
plot_logs(psf, ['lstm', 'UNNhirose', 'UNNmodRelu'], 'adding problem T=250',
          window_size=25, vtag='mse', ylim=[0.00, 0.35],
          tikz=True, filename='adding_problem_250_wisdom.tex')
plt.gcf().clear()

# p1 = base_path + '2018-05-04 03:45:00_memory_250_1000000_10000_128_0.001_50_LSTMCell'
# p2 = base_path + '2018-05-04 12:29:15_memory_250_1000000_10000_128_0.001_50_UnitaryCell__activation_hirose__arjovski_basis_False_nat_grad_rms_False_qr_steps_None'
# p3 = base_path + '2018-05-04 10:34:58_memory_250_1000000_10000_128_0.001_50_UnitaryCell__activation_mod_relu__arjovski_basis_False_nat_grad_rms_False_qr_steps_None'

# ps = [p1, p2, p3]
# psf = []
# for p in ps:
#     for _, _, files in os.walk(p):
#         psf.append(p + '/' + files[0])
# window_size = 25
# plot_logs(psf, ['lstm', 'UNNhirose', 'UNNmodRelu'], 'memory problem T=250',
#           window_size=25, vtag='cross_entropy', ylim=[0, .2],
#           tikz=True, filename='memory_problem_250_wisdom.tex')
