import numpy as np
from .dataset import Dataset

# single output = False
# num_in = 1
# num_target = 1

class TestProblemDataset(Dataset):
    def generate(self, num_samples):
        X = np.random.uniform(-1, 1, (num_samples, self.sample_len, 1))
        Y = np.zeros((num_samples, self.sample_len, 1))
        Y[:, range(0, self.sample_len, 3), 0] = 1
        return X, Y
