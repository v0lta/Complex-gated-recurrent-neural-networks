import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Audio
from intervaltree import Interval, IntervalTree

fs = 44100      # samples/second
train_data = np.load(open('../numpy/musicnet.npz', 'rb'),
                     encoding='latin1')

print('Number of recordings: ' + str(len(train_data.files)))
print('Example MusicNet ids: ' + str(train_data.keys()[0:5]))

X, Y = train_data['2494']
print(type(X))
print(type(Y))

if 0:
    fig = plt.figure()
    fig.set_figwidth(20)
    fig.set_figheight(2)
    plt.plot(X[0:30*fs], color=(41/255., 104/255., 168/255.))
    fig.axes[0].set_xlim([0, 30*fs])
    fig.axes[0].set_xlabel('sample (44,100Hz)')
    fig.axes[0].set_ylabel('amplitude')
    plt.show()

# Audio(X[0:30*fs], rate=fs)

print('Notes present at sample ' + str(fs*5) + ' (5 seconds): ' + str(len(Y[fs*5])))
print('Notes present at sample ' + str(fs*4) + ' (4 seconds): ' + str(len(Y[fs*4])))

(start, end, (instrument, note, measure, beat, note_value)) = sorted(Y[fs*5])[0]
print(' -- An example of a MusicNet label -- ')
print(' Start Time:                          ' + str(start))
print(' End Time:                            ' + str(end))
print(' Instrument (MIDI instrument code):   ' + str(instrument))
print(' Note (MIDI note code):               ' + str(note))
print(' Measure:                             ' + str(measure))
print(' Beat (0 <= beat < 1):                ' + str(beat))
print(' Note Value:                          ' + str(note_value))

# stride = 512                         # 512 samples between windows
stride = 1048
wps = fs/float(stride)               # ~86 windows/second
Yvec = np.zeros((int(30*wps), 128))  # 128 distinct note labels
colors = {41: .33, 42: .66, 43: 1}

for window in range(Yvec.shape[0]):
    labels = Y[window*stride]
    for label in labels:
        # label.data[1] encodes the note.
        Yvec[window, label.data[1]] = colors[label.data[0]]

fig = plt.figure(figsize=(20, 5))
plt.imshow(Yvec.T, aspect='auto', cmap='ocean_r')
plt.gca().invert_yaxis()
fig.axes[0].set_xlabel('window')
fig.axes[0].set_ylabel('note (MIDI code)')
plt.show()
