import numpy as np
from IPython.core.debugger import Tracer
debug_here = Tracer()

features_idx = 0    # first element of (X,Y) data tuple
labels_idx = 1      # second element of (X,Y) data tuple


class MusicNet(object):
    def __init__(self, c=24, stride=512, window_size=2048,
                 sampling_rate=44100, path='numpy/'):
        self._c = c
        self._stride = stride
        self._window_size = window_size
        self._m = 128
        if sampling_rate == 44100:
            path = path + 'musicnet.npz'
            self._fs = sampling_rate
            print('sr: ', sampling_rate, 'path: ', path)
        elif sampling_rate == 11000:
            path = path + 'musicnet11.npz'
            self._fs = sampling_rate
            print('sr: ', sampling_rate, 'path: ', path)
        else:
            raise ValueError("Sampling rate unavailable.")

        with open(path, 'rb') as data_file:
            self.train_data = dict(np.load(data_file, encoding='latin1'))
        print('musicnet loaded.')
        # split our the test set
        self.test_data = dict()
        for id in (2303, 2382, 1819):  # test set
            self.test_data[str(id)] = self.train_data.pop(str(id))

        self.train_ids = list(self.train_data.keys())
        self.test_ids = list(self.test_data.keys())

        print('splitting done.')
        print(len(self.train_data))
        print(len(self.test_data))
        assert window_size > 0

    # data selection funciton
    def select(self, data, index):
        time_music = []
        labels = []
        for cx in range(0, self._c):
                start = index - (self._c - cx)*self._stride
                center = start + int(self._window_size/2)
                end = start + self._window_size
                assert start >= 0.0
                time_data = data[features_idx][start:end]
                # label stuff that's on in the center of the window
                label_vec = np.zeros([self._m])
                for active_label in data[labels_idx][center]:
                    label_vec[active_label.data[1]] = 1
                time_music.append(time_data)
                labels.append(label_vec)

        if np.array(time_music).shape == (self._c, self._window_size) \
           and np.array(labels).shape == (self._c, self._m):
            pass
        else:
            print('incorrect array dimensions!')
            debug_here()
        return np.array(time_music), np.array(labels)

    def get_batch(self, data, data_indices, batch_size):
        """
        Get a training batch.
        Args:
            data: Dictionary {file_id, time_domain_numpy_array}
            data_indices: The file_id dictionary keys for data.
            batch_size: The batch size used in the graph.
        Returns:
            batch_time_music: (batch_size, c, d) array with time
                              domain data.
            batched_time_labels: (batch_size, c, m) array labels.
        """
        batch_time_music = []
        batched_time_labels = []
        batched = 0
        while batched < batch_size:
            # select a random recording from the data-set.
            dat_idx = np.random.randint(0, len(data_indices))
            # go to a random position in the recording.
            record_with_label = data[data_indices[dat_idx]]
            offset = self._window_size/2 + self._c*self._window_size
            rec_idx = np.random.randint(offset, len(record_with_label[features_idx])
                                        - self._window_size)
            time_music, labels = self.select(record_with_label, rec_idx)
            if time_music.shape == (self._c, self._window_size) \
               and labels.shape == (self._c, self._m):
                batch_time_music.append(time_music)
                batched_time_labels.append(labels)
                batched += 1
            else:
                pass
                # print('skipping sample.')

        batch_time_music = np.array(batch_time_music)
        batched_time_labels = np.array(batched_time_labels)

        # check the shapes.
        assert (batch_time_music.shape == (batch_size, self._c, self._window_size)
                and batched_time_labels.shape == (batch_size, self._c, self._m))
        return batch_time_music, batched_time_labels

    def get_test_batches(self, batch_size):
        """
        Set up the test set lists.
        """
        data = self.test_data
        data_indices = self.test_ids

        Xtest = []
        Ytest = []
        if self._fs == 44100:
            for dat_idx in data_indices:
                for j in range(7500):
                    if (1.0/self._fs)*self._window_size*self._c > 1.0:
                        rec_idx = self._window_size*self._c + j*512
                    else:
                        # start from one second to give us some room for larger segments
                        rec_idx = self._fs + j*512
                    record_with_label = data[dat_idx]
                    time_music, labels = self.select(record_with_label, rec_idx)
                    Xtest.append(time_music)
                    Ytest.append(labels)
            Xtest = np.array(Xtest)
            Ytest = np.array(Ytest)
        elif self._fs == 11000:
            for dat_idx in data_indices:
                for j in range(int(7500*(self._fs/44100.0))):
                    # make sure the is enough room for sample without wrap_around.
                    rec_idx = self._stride*self._c + j*int(512.0*(self._fs/44100.0))
                    record_with_label = data[dat_idx]
                    if rec_idx > record_with_label[features_idx].shape[0]:
                        print('stopping at', rec_idx,
                              record_with_label[features_idx].shape[0])
                        break
                    time_music, labels = self.select(record_with_label, rec_idx)
                    Xtest.append(time_music)
                    Ytest.append(labels)
            Xtest = np.array(Xtest)
            Ytest = np.array(Ytest)
        # Reshape and check the shapes.
        batched_music_lst = np.split(Xtest, int(Xtest.shape[0]/batch_size), axis=0)
        batcheded_labels_lst = np.split(Ytest, int(Xtest.shape[0]/batch_size), axis=0)
        assert len(batched_music_lst) == len(batcheded_labels_lst)
        return batched_music_lst, batcheded_labels_lst
