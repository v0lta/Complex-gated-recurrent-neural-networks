import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
from IPython.core.debugger import Tracer
debug_here = Tracer()


def tensoboard_average(y, window):
    '''
    * The smoothing algorithm is a simple moving average, which, given a
     * point p and a window w, replaces p with a simple average of the
     * points in the [p - floor(w/2), p + floor(w/2)] range.
    '''
    window_vals = []
    length = y.shape[-1]
    for p_no in range(0, length, window):
        if p_no > window/2 and p_no < length - window/2:
            window_vals.append(np.mean(y[p_no-int(window/2):p_no+int(window/2)]))
    return np.array(window_vals)


def plot_logs(ps, legend, title, window_size=25, vtag='mse', ylim=[0.00, 0.35],
              tikz=False, filename='tfplots.tex'):
    # cs = ['b', 'r', 'g']
    for no, p in enumerate(ps):
        adding_umc = []
        try:
            for e in tf.train.summary_iterator(p):
                for v in e.summary.value:
                    if v.tag == vtag:
                        # print(v.simple_value)
                        adding_umc.append(v.simple_value)
        except:
            # ingnore that silly data loss error....
            pass
        # x = np.array(range(len(adding_umc)))
        y = np.array(adding_umc)
        yhat = tensoboard_average(y, window_size)
        xhat = np.linspace(0, y.shape[0], yhat.shape[0])
        # plt.plot(yhat, cs[no])
        plt.plot(xhat, yhat, label=legend[no])

    plt.ylim(ylim[0], ylim[1])
    plt.grid()
    plt.ylabel(vtag)
    plt.xlabel('updates')
    plt.legend()
    plt.title(title)

    if tikz:
        from matplotlib2tikz import save as tikz_save
        tikz_save(filename)
    else:
        plt.show()

base_path = '/home/moritz/infcuda/complex_net_project/cgmu/phase-filter-RNN/logs/Loop_exp4/'
p1 = base_path + '2018-05-08 21:50:18_adding_250_1000000_10000_128_0.001_50_LSTMCell_pt_2560'
p2 = base_path + '2018-05-09 22:40:56_adding_250_1000000_10000_48_0.001_50_ComplexGatedRecurrentUnit__activation_hirose__arjovski_basis_False_state_U_True_inputFourier_True__inputHilbert_False__inputSplitMatmul_False__nat_grad_rms_False_qr_steps_None_pt_9890'
# p3 = base_path + '2018-05-09 22:46:13_adding_250_1000000_10000_48_0.001_50_ComplexGatedRecurrentUnit__activation_hirose__inputFourier_True__inputHilbert_False__inputSplitMatmul_False__weight_normalization_True_nat_grad_rms_False_qr_steps_None_pt_9890'
p4 = base_path + '2018-05-09 22:40:18_adding_250_1000000_10000_48_0.001_50_ComplexGatedRecurrentUnit__activation_hirose__inputFourier_True__inputHilbert_False__inputSplitMatmul_False__weight_normalization_False_nat_grad_rms_False_qr_steps_None_pt_9890'
ps = [p1, p2, p4]
psf = []
for p in ps:
    for _, _, files in os.walk(p):
        psf.append(p + '/' + files[0])
window_size = 25
plot_logs(psf, ['lstm', 'CGRU Unorm', 'CGRU free'], 'adding problem T=250',
          window_size=25, vtag='mse', ylim=[0.00, 0.35],
          tikz=True, filename='adding_problem_250_norm_no_norm_hirose.tex')
plt.gcf().clear()

p1 = base_path + '2018-05-08 22:59:07_memory_250_1000000_10000_128_0.001_50_LSTMCell_pt_7168'
p2 = base_path + '2018-05-09 08:10:31_memory_250_1000000_10000_48_0.001_50_ComplexGatedRecurrentUnit__activation_mod_relu__arjovski_basis_False_state_U_True_inputFourier_False__inputHilbert_False__inputSplitMatmul_False__nat_grad_rms_False_qr_steps_None_pt_10474'
# p3 = base_path + '2018-05-10 10:00:01_memory_250_1000000_10000_48_0.001_50_ComplexGatedRecurrentUnit__activation_mod_relu__inputFourier_True__inputHilbert_False__inputSplitMatmul_False__weight_normalization_True_nat_grad_rms_False_qr_steps_None_pt_10474'
p4 = base_path + '2018-05-10 10:09:01_memory_250_1000000_10000_48_0.001_50_ComplexGatedRecurrentUnit__activation_mod_relu__inputFourier_True__inputHilbert_False__inputSplitMatmul_False__weight_normalization_False_nat_grad_rms_False_qr_steps_None_pt_10474'

ps = [p1, p2, p4]
psf = []
for p in ps:
    for _, _, files in os.walk(p):
        psf.append(p + '/' + files[0])
window_size = 25
plot_logs(psf, ['lstm', 'CGRU Unorm', 'CGRU free'], 'memory problem T=250',
          window_size=25, vtag='cross_entropy', ylim=[0, .2],
          tikz=True, filename='memory_problem_250_norm_no_norm_mod_relu.tex')
