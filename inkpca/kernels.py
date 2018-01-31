
# -*- coding: utf-8 -*-

from itertools import combinations_with_replacement
from copy import deepcopy

import numpy as np
from numpy import ones, dot

def rbf(x, y, sigma=100):
    """
    Radial basis functions kernel

    """
    return np.exp(-(np.sum((x-y)**2))/(sigma**2))

def median_distance(X, n=1000):
    """
    Median distance between pairs of a subset of data examples

    """
    dist = np.zeros((n+1)*n/2)
    for k, t in enumerate(combinations_with_replacement(range(n),2)):
        dist[k] = np.sqrt(np.sum(np.power(X[t[0],:] - X[t[1],:],2)))
    sigma = np.median(dist)

    return sigma

def kernel_matrix(X, kernel, cols0, cols):
    """
    Calculate the kernel matrix

    """
    m = len(cols0)
    q = len(cols)
    K_mq = np.zeros((m,q))
    for i in range(m):
        for j in range(q):
            K_mq[i,j] = kernel(X[cols0[i]],X[cols[j]])

    return K_mq

def adjust_K(K):
    """
    Adjust a square kernel matrix to account for subtracted mean
    in feature space

    """
    m = K.shape[0]
    sums = np.expand_dims(K.sum(0),0) / m
    Kp = K - sums - sums.T + sums.sum() / m

    return Kp
