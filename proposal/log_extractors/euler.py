from helper_module import *
from IPython.core.debugger import Tracer
debug_here = Tracer()

const_path = '/home/moritz/infcuda/complex_net_project/human-motion-prediction/experiments/'
base_path = const_path + 'walking/out_25/iterations_20000/tied/sampling_based/one_hot/depth_1/size_1024/'

## learning rate 0.005
base_path += 'lr_0.005/not_residual_vel/log_complex/'

# Memory problem evaluation.
p1 = base_path + 'test/'
p2 = base_path + 'custom_opt/cgru/test/'
log = False
ylim = [1.0, 1.7]
title = 'euler error walking'
tikz = False
pdf = True
filename = 'euler.pdf'
ps = [p1, p2]
psf = []
for p in ps:
    for _, _, files in os.walk(p):
        psf.append(p + '/' + files[0])
window_size = 2
if not psf:
    print('psf is empty')
# plot_logs(psf, ['GRU', 'cGRU'],
#           'GRU, cGRU motion prediction',
#           window_size=window_size,
#           vtag='euler_error_walking/euler_error_walking/srnn_seeds_1000',
#           ylim=[1.15, 1.6],
#           tikz=False, pdf=True, filename='srnn_seeds_1000.pdf',
#           log=False)
# plt.gcf().clear()
vtags = ['euler_error_walking/euler_error_walking/srnn_seeds_0080',
         'euler_error_walking/euler_error_walking/srnn_seeds_0160',
         'euler_error_walking/euler_error_walking/srnn_seeds_0320',
         'euler_error_walking/euler_error_walking/srnn_seeds_0400',
         'euler_error_walking/euler_error_walking/srnn_seeds_0560',
         'euler_error_walking/euler_error_walking/srnn_seeds_1000']

# cs = ['b', 'r', 'g']
plt_lsts = []
for no, p in enumerate(psf):
    for vtag in vtags:
        adding_umc = []
        for e in tf.train.summary_iterator(p):
            for v in e.summary.value:
                if v.tag == vtag:
                    # print(v.simple_value)
                    adding_umc.append(v.simple_value)
        plt_lsts.append(adding_umc)

count = 0
for adding_umc in plt_lsts:
    y = np.array(adding_umc)
    # yhat = tensoboard_average(y, window_size)
    yhat = y
    xhat = np.linspace(0, y.shape[0], yhat.shape[0])
    # plt.plot(yhat, cs[no])
    if count < 6:
        if count == 0:
            plt.plot(xhat, yhat, 'b', label='GRU')
        else:
            plt.plot(xhat, yhat, 'b')
    else:
        if count == 6:
            plt.plot(xhat, yhat, 'g', label='cgRNN')
        else:
            plt.plot(xhat, yhat, 'g')
    count += 1

plt.ylim(ylim[0], ylim[1])
plt.grid()
plt.ylabel('euler error walking')
plt.xlabel('updates')
plt.legend()
plt.title(title)

if tikz:
    from matplotlib2tikz import save as tikz_save
    tikz_save(filename)
elif pdf:
    plt.savefig(filename, bbox_inches='tight')
else:
    plt.show()
