from helper_module import *
from IPython.core.debugger import Tracer
debug_here = Tracer()

# const_path = '/home/wolter/complex_net_project/cgmu/phase-filter-RNN/'
const_path = '/home/moritz/infcuda/complex_net_project/cgmu/phase-filter-RNN/'
base_path = const_path + 'logs/paper_final_real/'
bench_path = const_path + 'logs/paper_benchmarks/'

# Memory problem evaluation.
p1 = base_path + '2018-05-17 14:07:35_memory_250_1000000_10000_353_0.001_50_clipping_True_StiefelGatedRecurrentUnit__activation_relu__stiefel_True_real__nat_grad_rms_False_qr_steps_None_pt_387603'
p2 = base_path + '2018-05-17 14:07:08_memory_250_1000000_10000_353_0.001_50_clipping_True_StiefelGatedRecurrentUnit__activation_tanh__stiefel_True_real__nat_grad_rms_False_qr_steps_None_pt_387603'
p3 = base_path + '2018-05-17 14:10:36_memory_250_1000000_10000_353_0.001_50_clipping_True_StiefelGatedRecurrentUnit__activation_tanh__stiefel_False_real__nat_grad_rms_False_qr_steps_None_pt_387603'
p4 = base_path + '2018-05-17 19:27:09_memory_250_1000000_10000_353_0.001_50_clipping_True_StiefelGatedRecurrentUnit__activation_relu__stiefel_False_real__nat_grad_rms_False_qr_steps_None_pt_387603'

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
          'real networks, memory problem',
          window_size=25, vtag='cross_entropy', ylim=[0.00, 0.2],
          tikz=False, pdf=False, filename='memory_r.pdf',
          log=True)
plt.gcf().clear()
