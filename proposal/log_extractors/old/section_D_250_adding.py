#  gate comparison plots.

from helper_module import *
from IPython.core.debugger import Tracer
debug_here = Tracer()

const_path = '/home/moritz/infcuda/complex_net_project/cgmu/phase-filter-RNN/'
base_path = const_path + 'logs/paper_D/'
bench_path = const_path + 'logs/paper_benchmarks/'

p1 = bench_path + '2018-05-08 21:50:18_adding_250_1000000_10000_128_0.001_50_LSTMCell_pt_2560'
p2 = base_path + '2018-05-14 06:39:44_adding_250_1000000_10000_48_0.001_50_clipping_True_ComplexGatedRecurrentUnit__activation_mod_relu__input_fourier__stateU_True_gateO_False_singleGate_False_gate_activation_mod_sigmoid_prod_nat_grad_rms_False_qr_steps_None_pt_14786'
p3 = base_path + '2018-05-14 17:51:50_adding_250_1000000_10000_48_0.001_50_clipping_True_ComplexGatedRecurrentUnit__activation_mod_relu__input_fourier__stateU_True_gateO_False_singleGate_False_gate_activation_mod_sigmoid_beta_nat_grad_rms_False_qr_steps_None_pt_14790'
p4 = base_path + '2018-05-15 03:44:21_adding_250_1000000_10000_48_0.001_50_clipping_True_ComplexGatedRecurrentUnit__activation_mod_relu__input_fourier__stateU_True_gateO_False_singleGate_False_gate_activation_gate_phase_hirose_nat_grad_rms_False_qr_steps_None_pt_14792'
#p5 = base_path + '2018-05-13 18:07:00_adding_100_1000000_10000_48_0.001_50_clipping_True_ComplexGatedRecurrentUnit__activation_mod_relu__input_fourier__stateU_True_gateO_False_singleGate_False_gate_activation_mod_sigmoid_nat_grad_rms_False_qr_steps_None_pt_14788'
#ps = [p1, p2, p3, p4, p5]
ps = [p1, p2, p3, p4]
psf = []
for p in ps:
    for _, _, files in os.walk(p):
        psf.append(p + '/' + files[0])
window_size = 25
plot_logs(psf, ['lstm', 'prod sigmoid', 'mod sigoid beta', 'phase hirose'],
          'adding problem T=250, dual gate, mod relu',
          window_size=25, vtag='mse', ylim=[0.00, 0.25],
          tikz=True, filename='adding_problem_250_gate_comparison_mod_relu.tex')

# plot_logs(psf, ['lstm', 'phase hirose', 'prod sigmoid', 'mod sigoid beta', 'mod sigmoid'],
#           'adding problem T=100, dual gate, mod_relu',
#           window_size=25, vtag='mse', ylim=[0.00, 0.25],
#           tikz=False, filename='adding_problem_200_gate_comparison_mod_relu.tex')
plt.gcf().clear()


