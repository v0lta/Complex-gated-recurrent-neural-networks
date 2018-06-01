from helper_module import *
from IPython.core.debugger import Tracer
debug_here = Tracer()



# const_path = '/home/wolter/complex_net_project/cgmu/phase-filter-RNN/'
const_path = '/home/moritz/infcuda/complex_net_project/cgmu/phase-filter-RNN/'
bench_path = const_path + 'logs/paper_intro/'

# Intro memory problem.
p1 = bench_path + '2018-05-17 23:52:19_memory_250_1000000_10000_353_0.001_50_clipping_True_LSTMCell_pt_31064'
p2 = bench_path + '2018-05-18 02:10:00_memory_250_1000000_10000_250_0.001_50_clipping_True_UnitaryCell__activation_mod_relu__arjovski_basis_False__real_cell_False_nat_grad_rms_False_qr_steps_None_pt_134501'

ps = [p1, p2]
psf = []
for p in ps:
    for _, _, files in os.walk(p):
        psf.append(p + '/' + files[0])
window_size = 25
if not psf:
    print('psf is empty')
plot_logs(psf, ['LSTM', 'uRNN'],
          'Previous work LSTM and uRNN memory problem.',
          window_size=25, vtag='cross_entropy', ylim=[0.00, 0.2],
          tikz=False, pdf=True, filename='lstm_unn_intro_memory.pdf',
          log=False)

plt.gcf().clear()

# # Intro adding problem.
p1 = bench_path + '2018-05-17 23:50:21_adding_250_1000000_10000_128_0.001_50_clipping_True_LSTMCell_pt_2560'
p2 = bench_path + '2018-05-12 15:33:07_adding_250_1000000_10000_128_0.001_50_UnitaryCell__activation_mod_relu__arjovski_basis_False__real_cell_False_nat_grad_rms_False_qr_steps_None_pt_33794'

ps = [p1, p2]
psf = []
for p in ps:
    for _, _, files in os.walk(p):
        psf.append(p + '/' + files[0])
window_size = 25
if not psf:
    print('psf is empty')
plot_logs(psf, ['LSTM', 'uRNN mod_relu'],
          'Previous work LSTM and uRNN adding problem',
          window_size=25, vtag='mse', ylim=[0.00, 0.2],
          tikz=False, pdf=True, filename='lstm_unn_intro_adding.pdf',
          log=False)

plt.gcf().clear()
