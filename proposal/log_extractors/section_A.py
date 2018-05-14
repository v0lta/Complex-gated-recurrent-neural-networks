from helper_module import *
from IPython.core.debugger import Tracer
debug_here = Tracer()

const_path = '/home/moritz/infcuda/complex_net_project/cgmu/phase-filter-RNN/'
base_path = const_path + 'logs/paper_A/'
bench_path = const_path + 'logs/paper_benchmarks/'

p1 = bench_path + '2018-05-08 21:50:18_adding_250_1000000_10000_128_0.001_50_LSTMCell_pt_2560'
p2 = base_path + '2018-05-11 12:01:55_adding_250_1000000_10000_128_0.001_50_UnitaryCell__activation_relu__arjovski_basis_False__real_cell_True_nat_grad_rms_False_qr_steps_None_pt_16897/'
p3 = base_path + '2018-05-11 12:22:46_adding_250_1000000_10000_128_0.001_50_UnitaryCell__activation_mod_relu__arjovski_basis_False__real_cell_False_nat_grad_rms_False_qr_steps_None_pt_33794/'
ps = [p1, p2, p3]
psf = []
for p in ps:
    for _, _, files in os.walk(p):
        psf.append(p + '/' + files[0])
window_size = 25
plot_logs(psf, ['lstm', 'realUNNrelu', 'UNNmodRelu'], 'adding problem T=250, realUNN, UNN',
          window_size=25, vtag='mse', ylim=[0.00, 0.35],
          tikz=False, filename='adding_problem_250_R.vs.C.tex')
plt.gcf().clear()

### blow-up plots
# TODO.


p1 = bench_path + '2018-05-08 21:15:28_memory_100_1000000_10000_128_0.001_50_LSTMCell_pt_7168'
p2 = base_path + '2018-05-11 12:40:48_memory_250_1000000_10000_128_0.001_50_UnitaryCell__activation_relu__arjovski_basis_False__real_cell_True_nat_grad_rms_False_qr_steps_None_pt_17801/'
p3 = base_path + '2018-05-11 13:28:30_memory_250_1000000_10000_128_0.001_50_UnitaryCell__activation_mod_relu__arjovski_basis_False__real_cell_False_nat_grad_rms_False_qr_steps_None_pt_35594/'
ps = [p1, p2, p3]
psf = []
for p in ps:
    for _, _, files in os.walk(p):
        psf.append(p + '/' + files[0])
window_size = 25
plot_logs(psf, ['lstm', 'realUNN', 'UNN'], 'memory problem T=250, realUNN, UNN',
          window_size=25, vtag='cross_entropy', ylim=[0.00, 0.35],
          tikz=False, filename='adding_problem_T=250_R.vs.C.tex')
plt.gcf().clear()
