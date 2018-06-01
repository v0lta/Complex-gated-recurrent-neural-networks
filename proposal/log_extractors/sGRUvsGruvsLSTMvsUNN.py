

from helper_module import *
from IPython.core.debugger import Tracer
debug_here = Tracer()

# const_path = '/home/wolter/complex_net_project/cgmu/phase-filter-RNN/'
const_path = '/home/moritz/infcuda/complex_net_project/cgmu/phase-filter-RNN/'
base_path = const_path + 'logs/paper_memory128/'
bench_path = const_path + 'logs/paper_benchmarks/'

# Memory problem evaluation.
p1 = base_path + '2018-05-18 01:10:32_memory_250_1000000_10000_164_0.001_50_clipping_True_StiefelGatedRecurrentUnit__activation_mod_relu__stiefel_True_gate_activation_mod_sigmoid_prod_nat_grad_rms_False_qr_steps_None_pt_174169'
p2 = base_path + '2018-05-18 01:42:10_memory_250_1000000_10000_231_0.001_50_clipping_True_RealGRUWrapper_pt_169101'
p3 = base_path + '2018-05-18 04:14:57_memory_250_1000000_10000_200_0.001_50_clipping_True_LSTMCell_pt_17600'
p4 = base_path + '2018-05-18 06:08:42_memory_250_1000000_10000_286_0.001_50_clipping_True_UnitaryCell__activation_mod_relu__arjovski_basis_False__real_cell_False_nat_grad_rms_False_qr_steps_None_pt_174461'

ps = [p1, p2, p3, p4]
psf = []
for p in ps:
    for _, _, files in os.walk(p):
        psf.append(p + '/' + files[0])
window_size = 25
if not psf:
    print('psf is empty')
plot_logs(psf, ['cgRNN', 'GRU', 'LSTM', 'uRNN'],
          'memory problem',
          window_size=25, vtag='cross_entropy', ylim=[0.00, 0.2],
          tikz=False, pdf=True, filename='memory_sGRU_GRU_UNN_LSTM.pdf',
          log=False)
plt.gcf().clear()


# adding problem evaluation.
p1 = base_path + '2018-05-18 04:00:48_adding_250_1000000_10000_164_0.001_50_clipping_True_StiefelGatedRecurrentUnit__activation_mod_relu__stiefel_True_gate_activation_mod_sigmoid_prod_nat_grad_rms_False_qr_steps_None_pt_164657'
p2 = base_path + '2018-05-18 04:01:49_adding_250_1000000_10000_231_0.001_50_clipping_True_RealGRUWrapper_pt_162394'
p3 = base_path + '2018-05-18 04:11:30_adding_250_1000000_10000_800_0.001_50_clipping_True_LSTMCell_pt_16000'
p4 = base_path + '2018-05-18 06:08:22_adding_250_1000000_10000_286_0.001_50_clipping_True_UnitaryCell__activation_mod_relu__arjovski_basis_False__real_cell_False_nat_grad_rms_False_qr_steps_None_pt_165881'

ps = [p1, p2, p3, p4]
psf = []
for p in ps:
    for _, _, files in os.walk(p):
        psf.append(p + '/' + files[0])
window_size = 25
if not psf:
    print('psf is empty')
plot_logs(psf, ['cgRNN', 'GRU', 'LSTM', 'uRNN'],
          'adding problem',
          window_size=25, vtag='mse', ylim=[0.00, 0.2],
          tikz=False, pdf=True, filename='adding_sGRU_GRU_UNN_LSTM.pdf',
          log=False)
plt.gcf().clear()

