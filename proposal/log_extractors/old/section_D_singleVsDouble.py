from helper_module import *
from IPython.core.debugger import Tracer
debug_here = Tracer()

# const_path = '/home/wolter/complex_net_project/cgmu/phase-filter-RNN/'
const_path = '/home/moritz/infcuda/complex_net_project/cgmu/phase-filter-RNN/'
base_path = const_path + 'logs/paper_D/'
bench_path = const_path + 'logs/paper_benchmarks/'

# single vs. double hirose.
p1 = bench_path + '2018-05-08 21:50:18_adding_250_1000000_10000_128_0.001_50_LSTMCell_pt_2560'
p1 = base_path + '2018-05-09 22:40:56_adding_250_1000000_10000_48_0.001_50_ComplexGatedRecurrentUnit__activation_hirose__arjovski_basis_False_state_U_True_inputFourier_True__inputHilbert_False__inputSplitMatmul_False__nat_grad_rms_False_qr_steps_None_pt_9890'
p2 = base_path + '2018-05-15 09:35:09_adding_250_1000000_10000_39_0.001_50_clipping_True_ComplexGatedRecurrentUnit__activation_hirose__input_fourier__stateU_True_gateO_False_singleGate_False_gate_activation_mod_sigmoid_prod_nat_grad_rms_False_qr_steps_None_pt_9908'
#ps = [p1, p2, p3, p4, p5]
ps = [p1, p2]
psf = []
for p in ps:
    for _, _, files in os.walk(p):
        psf.append(p + '/' + files[0])
window_size = 25
if not psf:
    print('psf is empty')
plot_logs(psf, ['single', 'double'], 'adding problem T=250, single vs double, hirose',
          window_size=25, vtag='mse', ylim=[0.00, 0.2],
          tikz=True, filename='adding_singleVsDouble_hirose.tex')
plt.gcf().clear()

# single vs. double mod_relu.
p1 = bench_path + '2018-05-08 21:50:18_adding_250_1000000_10000_128_0.001_50_LSTMCell_pt_2560'
p1 = base_path + '2018-05-09 12:18:22_adding_250_1000000_10000_48_0.001_50_ComplexGatedRecurrentUnit__activation_mod_relu__arjovski_basis_False_state_U_True_inputFourier_True__inputHilbert_False__inputSplitMatmul_False__nat_grad_rms_False_qr_steps_None_pt_9890'
p2 = base_path + '2018-05-14 17:59:06_adding_250_1000000_10000_39_0.001_50_clipping_True_ComplexGatedRecurrentUnit__activation_mod_relu__input_fourier__stateU_True_gateO_False_singleGate_False_gate_activation_mod_sigmoid_prod_nat_grad_rms_False_qr_steps_None_pt_9908'
#ps = [p1, p2, p3, p4, p5]
ps = [p1, p2]
psf = []
for p in ps:
    for _, _, files in os.walk(p):
        psf.append(p + '/' + files[0])
window_size = 25
if not psf:
    print('psf is empty')
plot_logs(psf, ['single', 'double'], 'adding problem T=250, single vs double, mod relu',
          window_size=25, vtag='mse', ylim=[0.00, 0.2],
          tikz=True, filename='adding_singleVsDouble_mod_relu.tex')
plt.gcf().clear()
