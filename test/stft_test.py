import numpy as np
import functools
import scipy.signal as scisig
import tensorflow as tf
import tensorflow.contrib.signal as tfsig
from tensorflow.contrib.signal.python.ops import window_ops
import matplotlib.pyplot as plt
from IPython.core.debugger import Tracer
debug_here = Tracer()
tf.enable_eager_execution()


def istft(Zxx, win, fs=1.0, nperseg=None, noverlap=None, nfft=None,
          input_onesided=True, boundary=True, time_axis=-1, freq_axis=-2):
    Zxx = np.asarray(Zxx) + 0j
    freq_axis = int(freq_axis)
    time_axis = int(time_axis)

    nseg = Zxx.shape[time_axis]

    if input_onesided:
        # Assume even segment length
        n_default = 2*(Zxx.shape[freq_axis] - 1)
    else:
        n_default = Zxx.shape[freq_axis]

    if nfft is None:
        if (input_onesided) and (nperseg == n_default + 1):
            # Odd nperseg, no FFT padding
            nfft = nperseg
        else:
            nfft = n_default
    elif nfft < nperseg:
        raise ValueError('nfft must be greater than or equal to nperseg.')
    else:
        nfft = int(nfft)
    print('nfft', nfft)

    if noverlap is None:
        noverlap = nperseg//2
    else:
        noverlap = int(noverlap)
    if noverlap >= nperseg:
        raise ValueError('noverlap must be less than nperseg.')
    nstep = nperseg - noverlap
    print('nstep', nstep)
    print('noverlap', noverlap)

    # Rearrange axes if necessary
    if time_axis != Zxx.ndim-1 or freq_axis != Zxx.ndim-2:
        # Turn negative indices to positive for the call to transpose
        if freq_axis < 0:
            freq_axis = Zxx.ndim + freq_axis
        if time_axis < 0:
            time_axis = Zxx.ndim + time_axis
        zouter = list(range(Zxx.ndim))
        for ax in sorted([time_axis, freq_axis], reverse=True):
            zouter.pop(ax)
        Zxx = np.transpose(Zxx, zouter+[freq_axis, time_axis])

    if len(win.shape) != 1:
            raise ValueError('window must be 1-D')
    if win.shape[0] != nperseg:
            raise ValueError('window must have length of {0}'.format(nperseg))

    ifunc = np.fft.irfft

    xsubs = ifunc(Zxx, axis=-2, n=nfft)[..., :nperseg, :]

    # Initialize output and normalization arrays
    outputlength = nperseg + (nseg-1)*nstep
    x = np.zeros(list(Zxx.shape[:-2])+[outputlength], dtype=xsubs.dtype)
    norm = np.zeros(outputlength, dtype=xsubs.dtype)

    if np.result_type(win, xsubs) != xsubs.dtype:
        win = win.astype(xsubs.dtype)

    xsubs *= win.sum()  # This takes care of the 'spectrum' scaling

    # Construct the output from the ifft segments
    # This loop could perhaps be vectorized/strided somehow...
    for ii in range(nseg):
        # Window the ifft
        x[..., ii*nstep:ii*nstep+nperseg] += xsubs[..., ii] * win
        norm[..., ii*nstep:ii*nstep+nperseg] += win**2

    # Divide out normalization where non-tiny
    x /= np.where(norm > 1e-10, norm, 1.0)

    # Remove extension points
    if boundary:
        x = x[..., nperseg//2:-(nperseg//2)]

    if input_onesided:
        x = x.real

    # Put axes back
    if x.ndim > 1:
        if time_axis != Zxx.ndim-1:
            if freq_axis < time_axis:
                time_axis -= 1
            x = np.rollaxis(x, -1, time_axis)

    time = np.arange(x.shape[0])/float(fs)
    return time, x


fs = 10e3
N = 1e5
window_size = 1000
amp = 2 * np.sqrt(2)
noise_power = 0.01 * fs / 2
time = np.arange(N) / float(fs)
mod = 500*np.cos(2*np.pi*0.25*time)
carrier = amp * np.sin(2*np.pi*3e3*time + mod)
noise = np.random.normal(scale=np.sqrt(noise_power),
                         size=time.shape)
noise *= np.exp(-time/5)
x = carrier + noise
window = scisig.get_window('hann', window_size)
f, t, Zxx = scisig.stft(x, fs, nperseg=window_size, window=window)
# plt.pcolormesh(t, f, np.abs(Zxx), vmin=0, vmax=amp)
# plt.title('STFT Magnitude')
# plt.ylabel('Frequency [Hz]')
# plt.xlabel('Time [sec]')
# plt.show()
t, x_re = scisig.istft(Zxx, nperseg=window_size, window=window)
t_test, x_re_test = istft(Zxx, nperseg=window_size, win=window)
print('numpy reconstruction error:', np.linalg.norm(x - x_re))
print('numpy reconstruction error:', np.linalg.norm(x - x_re_test))


frame_length = 1000
frame_step = 250

center = True
if center:
    pad_amount = 2 * (frame_length - frame_step)
    x_pad = tf.pad(x.astype(np.float32), [[pad_amount // 2, pad_amount // 2]], 'REFLECT')
else:
    x_pad = x.astype(np.float32)

# f = tf.contrib.signal.frame(x_pad, frame_length, frame_step, pad_end=False)
# w = tf.contrib.signal.hann_window(frame_length, periodic=True)
# stfts = tf.spectral.rfft(f * w, fft_length=[frame_length])
# stfts = tf.spectral.rfft(f)
stfts = tf.contrib.signal.stft(x_pad, frame_length, frame_step)

# real_frames = tf.spectral.irfft(stfts)
# denom = tf.square(w)
# overlaps = -(-frame_length // frame_step)
# denom = tf.pad(denom, [(0, overlaps * frame_step - frame_length)])
# denom = tf.reshape(denom, [overlaps, frame_step])
# denom = tf.reduce_sum(denom, 0, keepdims=True)
# denom = tf.tile(denom, [overlaps, 1])
# denom = tf.reshape(denom, [overlaps * frame_step])
# w_inv = w / (denom)
# real_frames = real_frames*w_inv

# if center and pad_amount > 0:
#     real_frames = real_frames[pad_amount // 2:-pad_amount // 2]
output_T = tf.contrib.signal.inverse_stft(
    stfts, frame_length, frame_step,
    window_fn=tf.contrib.signal.inverse_stft_window_fn(frame_step))
if center and pad_amount > 0:
    output = output_T[pad_amount // 2:-pad_amount // 2]
else:
    output = output_T

output_array = np.array(output)
print(np.linalg.norm(x.astype(np.float32) - output_array))
plt.plot(x)
plt.plot(output_array)
# plt.plot(x - output_array)
plt.show()
