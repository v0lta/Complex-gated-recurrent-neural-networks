import numpy as np                                       # fast vectors and matrices
import matplotlib.pyplot as plt                          # plotting
from scipy import fft                                    # fast fourier transform
import tensorflow as tf
# tf.enable_eager_execution()
from IPython.display import Audio
from intervaltree import Interval, IntervalTree

fs = 44100      # samples/second
train_data = np.load(open('../numpy/musicnet.npz', 'rb'),
                     encoding='latin1')

X, Y = train_data['2494']  # data X and labels Y for recording id 1788

window_size = 2048  # 2048-sample fourier windows
stride = 512       # 512 samples between windows
wps = fs/float(512)  # ~86 windows/second
Xs = np.empty([int(10*wps), 2048])

for i in range(Xs.shape[0]):
    Xs[i] = np.abs(fft(X[i*stride:i*stride+window_size]))

second = 3

# fig, ((ax1, ax2, ax3)) = plt.subplots(1, 3, sharey=True)
# fig.set_figwidth(20)
# ax1.plot(Xs[int(second*wps)], color=(41/255., 104/255., 168/255.))
# ax1.set_xlim([0, window_size])
# ax1.set_ylabel('amplitude')
# ax2.plot(Xs[int(second*wps), 0:int(window_size/2)],
#          color=(41/255., 104/255., 168/255.))
# ax2.set_xlim([0, window_size/2])
# ax3.plot(Xs[int(second*wps), 0:150], color=(41/255.,
#                                             104/255., 168/255.))
# ax3.set_xlim([0, 150])
# plt.show()

# fig = plt.figure()
# fig.set_figwidth(20)
# fig.set_figheight(2)
# plt.plot(X[0:10*fs], color=(41/255., 104/255., 168/255.))
# fig.axes[0].set_xlabel('sample (44,100Hz)')
# fig.axes[0].set_ylabel('amplitude')
# plt.show()

if 0:
    fig = plt.figure(figsize=(20, 7))
    plt.imshow(Xs.T[0:150], aspect='auto')
    plt.gca().invert_yaxis()
    fig.axes[0].set_xlabel('windows (~86Hz)')
    fig.axes[0].set_ylabel('frequency')
    plt.show()

center = False
if center:
    pad_amount = 2 * (window_size - stride)
    x_pad = tf.pad(X.astype(np.float32).transpose(),
                   [[pad_amount // 2, pad_amount // 2]], 'REFLECT')
else:
    x_pad = X.astype(np.float32).transpose()

stfts = tf.contrib.signal.stft(x_pad, window_size, stride)

if 0:
    output_T = tf.contrib.signal.inverse_stft(
        stfts, window_size, stride,
        window_fn=tf.contrib.signal.inverse_stft_window_fn(stride))
    if center and pad_amount > 0:
        output = tf.transpose(output_T[pad_amount // 2:-pad_amount // 2])
    else:
        output = tf.transpose(output_T)

wps = fs/float(stride)                # ~86 windows/second
Yvec = np.zeros((int(10*wps), 128))   # 128 distinct note labels
colors = {41: .33, 42: .66, 43: 1}

for window in range(Yvec.shape[0]):
    labels = Y[window*stride]
    for label in labels:
        Yvec[window, label.data[1]] = colors[label.data[0]]


f, axarr = plt.subplots(1, 3)
axarr[0].imshow(Xs.T[0:150][::-1, :], aspect='auto')
axarr[1].imshow(np.abs(np.array(stfts[:Xs.shape[0], :]).transpose()[::-1, :][-150:, :]),
                aspect='auto')
axarr[2].imshow(Yvec.T, aspect='auto', cmap='ocean_r')
plt.gca().invert_yaxis()
plt.show()
