from helper_module import *
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
from IPython.core.debugger import Tracer
debug_here = Tracer()

const_path = '/home/moritz/infcuda/complex_net_project/cgmu/phase-filter-RNN/'
bench_path = const_path + 'logs/paper_benchmarks/'

base_path = '/home/moritz/infcuda/complex_net_project/cgmu/phase-filter-RNN/logs/Loop_exp4/'
p1 = bench_path + '2018-05-08 21:50:18_adding_250_1000000_10000_128_0.001_50_LSTMCell_pt_2560'
p2 = base_path + '2018-05-09 22:40:56_adding_250_1000000_10000_48_0.001_50_ComplexGatedRecurrentUnit__activation_hirose__arjovski_basis_False_state_U_True_inputFourier_True__inputHilbert_False__inputSplitMatmul_False__nat_grad_rms_False_qr_steps_None_pt_9890'
# p3 = base_path + '2018-05-09 22:46:13_adding_250_1000000_10000_48_0.001_50_ComplexGatedRecurrentUnit__activation_hirose__inputFourier_True__inputHilbert_False__inputSplitMatmul_False__weight_normalization_True_nat_grad_rms_False_qr_steps_None_pt_9890'
p4 = base_path + '2018-05-09 22:40:18_adding_250_1000000_10000_48_0.001_50_ComplexGatedRecurrentUnit__activation_hirose__inputFourier_True__inputHilbert_False__inputSplitMatmul_False__weight_normalization_False_nat_grad_rms_False_qr_steps_None_pt_9890'
ps = [p1, p2, p4]
psf = []
for p in ps:
    for _, _, files in os.walk(p):
        psf.append(p + '/' + files[0])
window_size = 25
plot_logs(psf, ['lstm', 'CGRU Unorm', 'CGRU free'], 'adding problem T=250',
          window_size=25, vtag='mse', ylim=[0.00, 0.35],
          tikz=False, filename='adding_problem_250_norm_no_norm_hirose.tex')
plt.gcf().clear()

# p1 = base_path + '2018-05-08 22:59:07_memory_250_1000000_10000_128_0.001_50_LSTMCell_pt_7168'
# p2 = base_path + '2018-05-09 08:10:31_memory_250_1000000_10000_48_0.001_50_ComplexGatedRecurrentUnit__activation_mod_relu__arjovski_basis_False_state_U_True_inputFourier_False__inputHilbert_False__inputSplitMatmul_False__nat_grad_rms_False_qr_steps_None_pt_10474'
# # p3 = base_path + '2018-05-10 10:00:01_memory_250_1000000_10000_48_0.001_50_ComplexGatedRecurrentUnit__activation_mod_relu__inputFourier_True__inputHilbert_False__inputSplitMatmul_False__weight_normalization_True_nat_grad_rms_False_qr_steps_None_pt_10474'
# p4 = base_path + '2018-05-10 10:09:01_memory_250_1000000_10000_48_0.001_50_ComplexGatedRecurrentUnit__activation_mod_relu__inputFourier_True__inputHilbert_False__inputSplitMatmul_False__weight_normalization_False_nat_grad_rms_False_qr_steps_None_pt_10474'

# ps = [p1, p2, p4]
# psf = []
# for p in ps:
#     for _, _, files in os.walk(p):
#         psf.append(p + '/' + files[0])
# window_size = 25
# plot_logs(psf, ['lstm', 'CGRU Unorm', 'CGRU free'], 'memory problem T=250',
#           window_size=25, vtag='cross_entropy', ylim=[0, .2],
#           tikz=True, filename='memory_problem_250_norm_no_norm_mod_relu.tex')
