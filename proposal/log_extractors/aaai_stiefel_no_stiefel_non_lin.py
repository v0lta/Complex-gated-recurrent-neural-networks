from helper_module import *
from IPython.core.debugger import Tracer
debug_here = Tracer()

# const_path = '/home/wolter/complex_net_project/cgmu/phase-filter-RNN/'
const_path = '/home/moritz/infcuda/complex_net_project/cgmu/phase-filter-RNN/'
base_path1 = const_path + 'logs/aaai_infcuda/'
base_path2 = const_path + 'logs/aaai_stiefel/'

# Memory problem evaluation.
p1 = base_path2 + '2018-06-08 14:10:24_memory_250_1000000_10000_80_0.001_50_clipping_True_StiefelGatedRecurrentUnit__activation_mod_relu__stiefel_True_gate_activation_mod_sigmoid_nat_grad_rms_False_qr_steps_None_pt_44643'
p2 = base_path2 + '2018-06-01 23:55:20_memory_250_1000000_10000_80_0.001_50_clipping_True_StiefelGatedRecurrentUnit__activation_hirose__stiefel_True_gate_activation_mod_sigmoid_nat_grad_rms_False_qr_steps_None_pt_44643'
p3 = base_path2 + '2018-06-01 20:09:57_memory_250_1000000_10000_80_0.001_50_clipping_True_StiefelGatedRecurrentUnit__activation_mod_relu__stiefel_False_gate_activation_mod_sigmoid_nat_grad_rms_False_qr_steps_None_pt_44643'
p4 = base_path2 + '2018-06-02 00:14:00_memory_250_1000000_10000_80_0.001_50_clipping_True_StiefelGatedRecurrentUnit__activation_hirose__stiefel_False_gate_activation_mod_sigmoid_nat_grad_rms_False_qr_steps_None_pt_44643'


ps = [p1, p2, p3, p4]
psf = []
for p in ps:
    for _, _, files in os.walk(p):
        psf.append(p + '/' + files[0])
window_size = 25
if not psf:
    print('psf is empty')
plot_logs(psf, ['stiefel mod_relu', 'stiefel hirose',
                'no stiefel mod_relu', 'no stiefel hirose'],
          'memory problem',
          window_size=25, vtag='cross_entropy', ylim=[0.00, 0.2],
          tikz=False, pdf=True, filename='aaai_stiefel_bounded_memory.pdf',
          log=False)
plt.gcf().clear()

# adding problem evaluation.
p1 = base_path2 + '2018-06-01 12:27:37_adding_250_1000000_10000_80_0.001_50_clipping_True_StiefelGatedRecurrentUnit__activation_mod_relu__stiefel_True_gate_activation_mod_sigmoid_nat_grad_rms_False_qr_steps_None_pt_40003'
p2 = base_path2 + '2018-06-01 16:10:49_adding_250_1000000_10000_80_0.001_50_clipping_True_StiefelGatedRecurrentUnit__activation_hirose__stiefel_True_gate_activation_mod_sigmoid_nat_grad_rms_False_qr_steps_None_pt_40003'
p3 = base_path2 + '2018-06-01 12:27:53_adding_250_1000000_10000_80_0.001_50_clipping_True_StiefelGatedRecurrentUnit__activation_mod_relu__stiefel_False_gate_activation_mod_sigmoid_nat_grad_rms_False_qr_steps_None_pt_40003'
p4 = base_path2 + '2018-06-01 16:17:56_adding_250_1000000_10000_80_0.001_50_clipping_True_StiefelGatedRecurrentUnit__activation_hirose__stiefel_False_gate_activation_mod_sigmoid_nat_grad_rms_False_qr_steps_None_pt_40003'

ps = [p1, p2, p3, p4]
psf = []
for p in ps:
    for _, _, files in os.walk(p):
        psf.append(p + '/' + files[0])
window_size = 25
if not psf:
    print('psf is empty')
plot_logs(psf, ['stiefel mod_relu', 'stiefel hirose',
                'no stiefel mod_relu', 'no stiefel hirose'],
          'adding problem',
          window_size=25, vtag='mse', ylim=[0.00, 0.2],
          tikz=False, pdf=True, filename='aaai_stiefel_bounded_adding.pdf',
          log=False)
plt.gcf().clear()
