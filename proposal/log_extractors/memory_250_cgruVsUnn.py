from helper_module import *
from IPython.core.debugger import Tracer
debug_here = Tracer()

const_path = '/home/moritz/infcuda/complex_net_project/cgmu/phase-filter-RNN/'
base_path = const_path + 'logs/paperA_2/'
bench_path = const_path + 'logs/paper_benchmarks/'

p1 = base_path + '2018-05-15 11:47:03_memory_250_1000000_10000_256_0.001_50_clipping_True_LSTMCell_pt_22528'
p2 = base_path + '2018-05-15 15:47:52_memory_250_1000000_10000_128_0.001_50_clipping_True_ComplexGatedRecurrentUnit__activation_mod_relu__stateU_False_gateO_False_singleGate_True_single_gate_avg_False_nat_grad_rms_False_qr_steps_None_pt_72970'
p3 = base_path + '2018-05-14 22:24:28_memory_250_1000000_10000_128_0.001_50_clipping_True_ComplexGatedRecurrentUnit__activation_mod_relu__stateU_True_gateO_False_singleGate_True_single_gate_avg_False_nat_grad_rms_False_qr_steps_None_pt_72970'
p4 = base_path + '2018-05-15 11:29:21_memory_250_1000000_10000_128_0.001_50_clipping_True_UnitaryCell__activation_mod_relu__arjovski_basis_False__real_cell_False_nat_grad_rms_False_qr_steps_None_pt_37642'

ps = [p1, p2, p3, p4]
psf = []
for p in ps:
    for _, _, files in os.walk(p):
        psf.append(p + '/' + files[0])
window_size = 25
plot_logs(psf, ['LSTM', 'CGRU free', 'CGRU', 'UNN'],
          'memory problem T=250, UNN vs. CGRU',
          window_size=25, vtag='cross_entropy', ylim=[0.00000001, 1.3],
          tikz=False, filename='memory_problem_250_unnVsCGRU.tex', log=True)
