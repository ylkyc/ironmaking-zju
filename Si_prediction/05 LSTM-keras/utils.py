import numpy as np
import pickle

def maxmin(x, x_max, x_min):
    """[0,1] normaliaztion"""
    x = (x - x_min) / (x_max - x_min)
    return x


def reverse_normalization(y, yy_max, yy_min):
    y = y.reshape(-1)
    reverse_y = y * (yy_max - yy_min) + yy_min
    return  reverse_y


def time_window(dataset, width):
    features = []
    labels = []
    for i in range(dataset.shape[0] - width):
        features_series = dataset[i:width + i, :-1]
        label = dataset[width + i - 1, -1]
        features.append(features_series)
        labels.append(label)

    features = np.array(features)
    labels = np.array(labels)
    return features, labels


def divide(data, train_size, valid_size, test_size):
    data_train = data[:train_size]
    data_valid = data[train_size:train_size + valid_size]
    data_test = data[train_size + valid_size:train_size + valid_size + test_size]
    return data_train, data_valid, data_test


def load_timeseries(dataset, width, train_size, valid_size, test_size):
    train_dataset, valid_dataset, test_dataset = divide(dataset, train_size, test_size, valid_size)
    dataset_min = np.min(train_dataset)
    dataset_max = np.max(train_dataset)

    train_dataset_normal = np.array(maxmin(train_dataset, dataset_max, dataset_min))
    valid_dataset_normal = np.array(maxmin(valid_dataset, dataset_max, dataset_min))
    test_dataset_normal = np.array(maxmin(test_dataset, dataset_max, dataset_min))

    x_train, y_train = time_window(train_dataset_normal, width)
    x_valid, y_valid = time_window(valid_dataset_normal, width)
    x_test, y_test = time_window(test_dataset_normal, width)

    return x_train, x_valid, x_test, y_train, y_valid, y_test, dataset_min, dataset_max


class DataSet(object):
    def __init__(self, x):
        self._data_size = x.shape[0]
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self._data_index = np.arange(x.shape[0])
        self.x = x

    def next_batch(self, batch_size):
        start = self._index_in_epoch
        if start + batch_size > self._data_size:
            self._index_in_epoch = 0
            start = self._index_in_epoch
            end = self._index_in_epoch + batch_size
            self._index_in_epoch = end
        else:
            end = self._index_in_epoch + batch_size
            self._index_in_epoch = end
        batch_x = self.get_data(start, end)
        return np.array(batch_x, dtype=np.float32)

    def get_data(self, start, end):
        batch_x = []
        for i in range(start, end):
            batch_x.append(self.x[self._data_index[i]])
        return batch_x