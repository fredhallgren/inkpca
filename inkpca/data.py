
# -*- coding: utf-8 -*-

import numpy as np


def get_magic_data():
    """
    """
    f = open("../data/magic_gamma_telescope")
    data = []
    for line in f.readlines():
        line = line.split(',')[:-1]
        data.append(line)
    data = np.asarray(data, dtype=np.float64)

    return data

def get_yeast_data():
    """
    """
    f = open("../data/yeast")
    data = []
    for line in f.readlines():
        line = line.split('  ')
        line = line[1:-1] # keep float values
        try:
            np.asarray(line, dtype=np.float64)
        except:
            continue
        data.append(line)
    data = np.asarray(data, dtype=np.float64)

    return data

