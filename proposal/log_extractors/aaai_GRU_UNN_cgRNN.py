from helper_module import *
from IPython.core.debugger import Tracer
debug_here = Tracer()

# const_path = '/home/wolter/complex_net_project/cgmu/phase-filter-RNN/'
const_path = '/home/moritz/infcuda/complex_net_project/cgmu/phase-filter-RNN/'
base_path1 = const_path + 'logs/aaai_infcuda/'
base_path2 = const_path + 'logs/aaai_auersberg/'

# Memory problem evaluation.
p1 = base_path2 + '2018-05-31 15:49:55_memory_250_1000000_10000_140_0.001_50_clipping_True_UnitaryCell__activation_mod_relu__arjovski_basis_False__real_cell_False_nat_grad_rms_False_qr_steps_None_pt_44521'
p2 = base_path2 + '2018-05-25 17:48:16_memory_250_1000000_10000_112_0.001_50_clipping_True_RealGRUWrapper_pt_42009'
p3 = base_path1 + '2018-05-26 00:55:31_memory_250_1000000_10000_80_0.001_50_clipping_True_StiefelGatedRecurrentUnit__activation_mod_relu__stiefel_True_gate_activation_mod_sigmoid_nat_grad_rms_False_qr_steps_None_pt_44643'

ps = [p1, p2, p3]
psf = []
for p in ps:
    for _, _, files in os.walk(p):
        psf.append(p + '/' + files[0])
window_size = 25
if not psf:
    print('psf is empty')
plot_logs(psf, ['uRNN', 'GRU', 'cgRNN'],
          'memory problem',
          window_size=25, vtag='cross_entropy', ylim=[0.00, 0.2],
          tikz=False, pdf=True, filename='aaai_GRU_UNN_cgRNN_memory.pdf',
          log=False)
plt.gcf().clear()

# adding problem evaluation.
p1 = base_path2 + '2018-05-31 15:50:29_adding_250_1000000_10000_140_0.001_50_clipping_True_UnitaryCell__activation_mod_relu__arjovski_basis_False__real_cell_False_nat_grad_rms_False_qr_steps_None_pt_40321'
p2 = base_path2 + '2018-05-25 15:55:26_adding_250_1000000_10000_112_0.001_50_clipping_True_RealGRUWrapper_pt_38753'
p3 = base_path1 + '2018-05-25 16:13:58_adding_250_1000000_10000_80_0.001_50_clipping_True_StiefelGatedRecurrentUnit__activation_mod_relu__stiefel_True_gate_activation_mod_sigmoid_nat_grad_rms_False_qr_steps_None_pt_40003'

ps = [p1, p2, p3]
psf = []
for p in ps:
    for _, _, files in os.walk(p):
        psf.append(p + '/' + files[0])
window_size = 25
if not psf:
    print('psf is empty')
plot_logs(psf, ['uRNN', 'GRU', 'cgRNN'],
          'adding problem',
          window_size=25, vtag='mse', ylim=[0.00, 0.2],
          tikz=False, pdf=True, filename='aaai_GRU_UNN_cgRNN_adding.pdf',
          log=False)
plt.gcf().clear()
