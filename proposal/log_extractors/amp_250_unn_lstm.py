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
        for e in tf.train.summary_iterator(p):
            for v in e.summary.value:
                if v.tag == vtag:
                    # print(v.simple_value)
                    adding_umc.append(v.simple_value)
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

base_path = '/home/moritz/infcuda/complex_net_project/cgmu/phase-filter-RNN/logs/Loop_exp3/'
p1 = base_path + '2018-05-03 23:21:38_adding_250_1000000_10000_128_0.001_50_LSTMCell'
p2 = base_path + '2018-05-04 05:25:32_adding_250_1000000_10000_128_0.001_50_UnitaryCell__activation_hirose__arjovski_basis_False_nat_grad_rms_False_qr_steps_None'
p3 = base_path + '2018-05-04 03:33:18_adding_250_1000000_10000_128_0.001_50_UnitaryCell__activation_mod_relu__arjovski_basis_False_nat_grad_rms_False_qr_steps_None'
p4 = base_path + ''
ps = [p1, p2, p3]
psf = []
for p in ps:
    for _, _, files in os.walk(p):
        psf.append(p + '/' + files[0])
window_size = 25
plot_logs(psf, ['lstm', 'UNNhirose', 'UNNmodRelu'], 'adding problem T=250',
          window_size=25, vtag='mse', ylim=[0.00, 0.35],
          tikz=True, filename='adding_problem_250_wisdom.tex')
plt.gcf().clear()

p1 = base_path + '2018-05-04 03:45:00_memory_250_1000000_10000_128_0.001_50_LSTMCell'
p2 = base_path + '2018-05-04 12:29:15_memory_250_1000000_10000_128_0.001_50_UnitaryCell__activation_hirose__arjovski_basis_False_nat_grad_rms_False_qr_steps_None'
p3 = base_path + '2018-05-04 10:34:58_memory_250_1000000_10000_128_0.001_50_UnitaryCell__activation_mod_relu__arjovski_basis_False_nat_grad_rms_False_qr_steps_None'

ps = [p1, p2, p3]
psf = []
for p in ps:
    for _, _, files in os.walk(p):
        psf.append(p + '/' + files[0])
window_size = 25
plot_logs(psf, ['lstm', 'UNNhirose', 'UNNmodRelu'], 'memory problem T=250',
          window_size=25, vtag='cross_entropy', ylim=[0, .2],
          tikz=True, filename='memory_problem_250_wisdom.tex')
