from helper_module import *
from IPython.core.debugger import Tracer
debug_here = Tracer()

const_path = '/home/moritz/infcuda/complex_net_project/cgmu/phase-filter-RNN/'
base_path = const_path + 'logs/paper_D/'
bench_path = const_path + 'logs/paper_benchmarks/'

p1 = bench_path + '2018-05-08 21:50:18_adding_250_1000000_10000_128_0.001_50_LSTMCell_pt_2560/'
p2 = base_path + '2018-05-11 21:06:11_adding_250_1000000_10000_48_0.001_50_UnitaryMemoryCell__activation_mod_relu_input_fourier__singleGate_False__nat_grad_rms_False_qr_steps_None_pt_5478/'
p3 = base_path + '2018-05-11 23:21:09_adding_250_1000000_10000_48_0.001_50_UnitaryMemoryCell__activation_hirose_input_fourier__singleGate_False__nat_grad_rms_False_qr_steps_None_pt_5478/'
ps = [p1, p2, p3]
psf = []
for p in ps:
    for _, _, files in os.walk(p):
        psf.append(p + '/' + files[0])
window_size = 25
print(psf)
plot_logs(psf, ['lstm', 'GUNNmodRelu', 'GUNNhirose'], 'adding problem T=250, GUNN single_gate',
          window_size=25, vtag='mse', ylim=[0.00, 0.35],
          tikz=False, filename='adding_problem_250_gunn_single.tex')
plt.gcf().clear()

p1 = bench_path + '2018-05-08 22:59:07_memory_250_1000000_10000_128_0.001_50_LSTMCell_pt_7168'
p2 = base_path + '2018-05-12 01:37:20_memory_250_1000000_10000_48_0.001_50_UnitaryMemoryCell__activation_mod_relu_input_fourier__singleGate_False__nat_grad_rms_False_qr_steps_None_pt_5966'
p3 = base_path + '2018-05-12 04:01:20_memory_250_1000000_10000_48_0.001_50_UnitaryMemoryCell__activation_hirose_input_fourier__singleGate_False__nat_grad_rms_False_qr_steps_None_pt_5966'
ps = [p1, p2, p3]
psf = []
for p in ps:
    for _, _, files in os.walk(p):
        psf.append(p + '/' + files[0])
window_size = 25
plot_logs(psf, ['lstm', 'GUNNmodRelu', 'GUNNhirose'], 'memory problem T=250, GUNN single_gate',
          window_size=25, vtag='cross_entropy', ylim=[0.00, 0.35],
          tikz=False, filename='memory_250_gunn_single.tex')
plt.gcf().clear()
