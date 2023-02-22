import numpy as np


class DataLoader(object):
    """
    Tool for shuffling data and forming mini-batches
    """
    def __init__(self, X, y, batch_size=1, shuffle=False):
        """
        :param X: dataset features
        :param y: dataset targets
        :param batch_size: size of mini-batch to form
        :param shuffle: whether to shuffle dataset
        """
        assert X.shape[0] == y.shape[0]
        self.X = X
        self.y = y
        self.data = np.c_[X, y]
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.batch_id = 0

    def __len__(self) -> int:
        """
        :return: number of batches per epoch
        """
        return (self.X.shape[0] + self.batch_size - 1) // self.batch_size

    def num_samples(self) -> int:
        """
        :return: number of data samples
        """
        return self.X.shape[0]

    def __iter__(self):
        """
        Shuffle data samples if required
        :return: self
        """
        self.batch_id = 0
        if self.shuffle:
            np.random.shuffle(self.data)
            self.X = self.data[:, :-1].reshape(self.X.shape)
            self.y = self.data[:, -1].reshape(self.y.shape)
        return self

    def __next__(self):
        """
        Form and return next data batch
        :return: (x_batch, y_batch)
        """
        if self.batch_id < self.__len__():
            low = self.batch_id * self.batch_size
            high = min(low + self.batch_size, self.num_samples())
            
            self.batch_id += 1
            
            return self.X[low:high, :], self.y[low:high]
        
        raise StopIteration
