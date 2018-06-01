from helper_module import *
from IPython.core.debugger import Tracer
debug_here = Tracer()

const_path = '/home/moritz/infcuda/complex_net_project/cgmu/phase-filter-RNN/'
base_path = const_path + 'logs/paper_D/'
bench_path = const_path + 'logs/paper_benchmarks/'

p1 = bench_path + '2018-05-08 20:45:19_adding_100_1000000_10000_128_0.001_50_LSTMCell_pt_2560/'
p2 = base_path + '2018-05-11 16:59:38_adding_100_1000000_10000_48_0.001_50_ComplexGatedRecurrentUnit__activation_mod_relu__input_fourier__stateU_True_gateO_False_singleGate_False_nat_grad_rms_False_qr_steps_None_pt_14790/'
p3 = base_path + '2018-05-11 22:45:26_adding_100_1000000_10000_48_0.001_50_ComplexGatedRecurrentUnit__activation_hirose__input_fourier__stateU_True_gateO_False_singleGate_False_nat_grad_rms_False_qr_steps_None_pt_14790/'
ps = [p1, p2, p3]
psf = []
for p in ps:
    for _, _, files in os.walk(p):
        psf.append(p + '/' + files[0])
window_size = 25
print(psf)
plot_logs(psf, ['lstm', 'CGRUmodRelu', 'CGRUhirose'], 'adding problem T=100, CGRU double_gate',
          window_size=25, vtag='mse', ylim=[0.00, 0.35],
          tikz=False, filename='adding_problem_100_CGRU_double_gate.tex')
plt.gcf().clear()

p1 = bench_path + '2018-05-08 21:15:28_memory_100_1000000_10000_128_0.001_50_LSTMCell_pt_7168'
p2 = base_path + '2018-05-12 04:39:24_memory_100_1000000_10000_48_0.001_50_ComplexGatedRecurrentUnit__activation_mod_relu__input_fourier__stateU_True_gateO_False_singleGate_False_nat_grad_rms_False_qr_steps_None_pt_15278'
p3 = base_path + '2018-05-12 11:50:42_memory_100_1000000_10000_48_0.001_50_ComplexGatedRecurrentUnit__activation_hirose__input_fourier__stateU_True_gateO_False_singleGate_False_nat_grad_rms_False_qr_steps_None_pt_15278'
ps = [p1, p2, p3]
psf = []
for p in ps:
    for _, _, files in os.walk(p):
        psf.append(p + '/' + files[0])
window_size = 25
plot_logs(psf, ['lstm', 'CGRUmodRelu', 'CGRUhirose'], 'memory problem T=100, CGRU double_gate',
          window_size=25, vtag='cross_entropy', ylim=[0.00, 0.35],
          tikz=False, filename='memory_100_CGRU_double_gate.tex')
plt.gcf().clear()
