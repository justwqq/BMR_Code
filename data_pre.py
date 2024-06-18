############Data Preparation############
import scipy.io
import numpy as np
import torch
class DATASET(object):
    def read_data(self):
        site1 = scipy.io.loadmat('/media/qqw/Elements/qqw/A_PycharmProjects/mywork_demo/abide.mat')
        A = np.squeeze(site1['A'].T)
        series = []
        for i in range(len(A)):
            signal = A[i]
            series.append(signal)
        X = np.array(series)
        y = np.squeeze(site1['label'])
        return X, y

    def __init__(self):
        super(DATASET, self).__init__()
        X, y = self.read_data()
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)
        self.n_samples = X.shape[0]

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        return self.X[index], self.y[index]
