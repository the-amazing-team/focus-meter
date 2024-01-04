from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
import numpy as np


def one_hot_encode(labels):
    encoder = LabelEncoder()
    labels = np_utils.to_categorical(encoder.fit_transform(labels))
    return labels


def auto_pad(arr):
    max_len = max(len(lst) for lst in arr)
    return np.array([np.pad(lst, (0, max_len - len(lst)), "constant") for lst in arr])
