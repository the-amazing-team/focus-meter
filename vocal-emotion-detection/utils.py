import numpy as np


def auto_pad(arr):
    max_len = max(len(lst) for lst in arr)
    return np.array([np.pad(lst, (0, max_len - len(lst)), "constant") for lst in arr])
