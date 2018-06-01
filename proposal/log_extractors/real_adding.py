from helper_module import *
from IPython.core.debugger import Tracer
debug_here = Tracer()

# const_path = '/home/wolter/complex_net_project/cgmu/phase-filter-RNN/'
const_path = '/home/moritz/infcuda/complex_net_project/cgmu/phase-filter-RNN/'
base_path = const_path + 'logs/paper_final_real/'
bench_path = const_path + 'logs/paper_benchmarks/'

# Memory problem evaluation.
p1 = base_path + '2018-05-17 15:37:18_adding_250_1000000_10000_128_0.001_50_clipping_True_StiefelGatedRecurrentUnit__activation_relu__stiefel_True_real__nat_grad_rms_False_qr_steps_None_pt_50433'
p2 = base_path + '2018-05-17 15:36:40_adding_250_1000000_10000_128_0.001_50_clipping_True_StiefelGatedRecurrentUnit__activation_tanh__stiefel_True_real__nat_grad_rms_False_qr_steps_None_pt_50433'
p3 = base_path + '2018-05-17 15:41:11_adding_250_1000000_10000_128_0.001_50_clipping_True_StiefelGatedRecurrentUnit__activation_tanh__stiefel_False_real__nat_grad_rms_False_qr_steps_None_pt_50433'
p4 = base_path + '2018-05-17 15:42:35_adding_250_1000000_10000_128_0.001_50_clipping_True_StiefelGatedRecurrentUnit__activation_relu__stiefel_False_real__nat_grad_rms_False_qr_steps_None_pt_50433'


ps = [p1, p2, p3, p4]
psf = []
for p in ps:
    for _, _, files in os.walk(p):
        psf.append(p + '/' + files[0])
window_size = 25
if not psf:
    print('psf is empty')
plot_logs(psf, ['stiefel relu', 'stiefel tanh',
                'no stiefel tanh', 'no stiefel relu'],
          'real networks, adding problem',
          window_size=25, vtag='mse', ylim=[0.00, 0.2],
          tikz=False, pdf=True, filename='adding_r.pdf',
          log=False)
plt.gcf().clear()
