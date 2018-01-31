
# -*- coding: utf-8 -*-

from __future__ import division, print_function

# This package
import data
from incremental_kpca import IncrKPCA, nystrom_approximation
from kernels import kernel_matrix, rbf, adjust_K, median_distance
from chinsuter import ChinSuter

# Built-in modules
import sys
from time import time

# External modules
import numpy as np
from numpy import dot, diag
from matplotlib import pyplot as plt
from matplotlib import rcParams

# Matplotlib config
rcParams['font.family'] = 'serif'
rcParams['axes.titlesize'] = 21
rcParams['axes.labelsize'] = 19
rcParams['xtick.labelsize'] = 15
rcParams['ytick.labelsize'] = 15


def main(dataset='magic', datasize=1000):
    """
    Run experiments for the incremental kernel PCA algorithm and the
    incremental Nyström approximation.

    After each plot is shown the program halts, close the plot to continue.

    Parameters
    ----------
    dataset : str
        Either 'magic' or 'yeast'
    datasize : int or None
        Size of dataset for Nyström comparison

    """

    if not dataset in ('magic', 'yeast'):
        raise ValueError("Unknown dataset.")

    X = getattr(data, "get_" + dataset + "_data")()

    if datasize:
        Xcut = X[:datasize]

    sigma = median_distance(X)

    kernel = lambda x, y: rbf(x, y, sigma)

    mmax = 100

    m0 = 20

    incremental_experiment(X, m0, mmax, kernel, dataset)

    incremental_experiment(X, m0, mmax, kernel, dataset, adjust=True)

    nystrom_experiment(Xcut, m0, mmax, kernel, dataset)


def incremental_experiment(X, m0, mmax, kernel, dataset, adjust=False):
    """
    Experiment for the incremental kernel pca algorithm. For each additional
    data point the difference in Frobenius norm between incremental and batch
    calculation is plotted (termed drift).

    Parameters
    ----------

    X : numpy.ndarray, 2d
        Data matrix
    m0 : int
        Initial size of kernel matrix
    mmax : int
        Maximum size of kernel matrix
    kernel : callable
        Kernel function
    dataset : str
        Either 'magic' or 'yeast'
    adjust : bool
        Whether to adjust the mean

    """
    print("\nIncremental kernel PCA")
    inc = IncrKPCA(X, m0, mmax, adjust=adjust, kernel=kernel)
    fnorms = []
    for i, L, U in inc:
        idx = inc.get_idx_array()
        K = kernel_matrix(X, kernel, idx[:i+1], idx[:i+1])
        if adjust:
            K = adjust_K(K)
        K_tilde = dot(U, dot(diag(L), U.T))
        fnorm = np.sqrt(np.sum(np.sum(np.power(K - K_tilde, 2))))
        fnorms.append(fnorm)

    plotting(np.arange(len(fnorms))+m0, fnorms, dataset, "m", "Frobenius norm")

def nystrom_experiment(X, m0, mmax, kernel, dataset):
    """
    Incremental calculation of the Nyström approximation to the kernel matrix.
    For each data point the difference in Frobenius norm between the
    approximation and the full kernel matrix is plotted.

    Parameters
    ----------

    X : numpy.ndarray, 2d
        Data matrix
    m0 : int
        Initial size of Nyström subset
    mmax : int
        Maximum size of Nyström subset
    kernel : callable
        Kernel function
    dataset : str
        Either 'magic' or 'yeast'

    """
    print("\nIncremental Nyström approximation")
    inc = IncrKPCA(X, m0, mmax, kernel=kernel, nystrom=True)
    idx = inc.get_idx_array()
    n = X.shape[0]
    K = kernel_matrix(X, kernel, range(n), range(n))
    fnorms = []
    for i, L, U, L_nys, U_nys in inc:
        K_tilde = dot(U_nys, dot(diag(L_nys), U_nys.T))
        fnorm = np.sqrt(np.sum(np.sum(np.power(K - K_tilde, 2))))
        fnorms.append(fnorm)

    plotting(range(m0, m0+len(fnorms)), fnorms, dataset, "m", "Frobenius norm")


def plotting(x, y, title, xlabel, ylabel):
    plt.figure()
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main(*sys.argv[1:])
