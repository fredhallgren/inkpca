
from data import get_magic_data
from kernels import kernel_matrix, kernel_matrices, rbf
from incremental_kpca import nystrom_approximation
from linalg import *

import numpy as np
from numpy import eye, zeros, dot, diag
from scipy import linalg
from itertools import product
from nose.tools import assert_almost_equal, assert_less, set_trace


def test_rank_one_eigenvalues():
    L = np.arange(1,4)
    A = np.diag(L)
    z = np.arange(1,4) / 5.0
    z = np.expand_dims(z,1)
    sigma = 1
    A2 = A + sigma * dot(z,z.T)
    exp, _ = linalg.eigh(A2)
    act = rank_one_eigenvalues(L, z, sigma)
    for i in product(range(3)):
        assert_almost_equal(exp[i], act[i])

def test_rank_one_eigenvectors():
    A = np.array([[1, 1, 1],
                  [1, 2, 1],
                  [1, 1, 3]])
    L, U = linalg.eigh(A)
    v = np.arange(1,4)
    v = np.expand_dims(v,1)
    z = dot(U.T,v)
    sigma = 1
    A2 = A + sigma * dot(v,v.T)
    L_exp, U_exp = linalg.eigh(A2)
    U_act = rank_one_eigenvectors(L, L_exp, U, z)
    for i,j in product(range(3),range(3)):
        assert_almost_equal(np.abs(U_exp[i,j]), np.abs(U_act[i,j]))

def test_nystrom_approx():
    datasize=100
    fraction=0.1

    X = get_magic_data()
    X = X[:datasize]
    cols = range(int(fraction*datasize))
    all_cols = range(datasize)

    K_mm, K_nm = kernel_matrices(X, rbf, cols)
    K = kernel_matrix(X, rbf, all_cols, all_cols)
    L, U = linalg.eigh(K_mm)
    L_nys, U_nys = nystrom_approximation(L, U, K_nm)
    K_nys = dot(U_nys, dot(diag(L_nys), U_nys.T))

    # F norm of difference
    fnorm = np.sqrt(np.sum(np.sum(np.power(K - K_nys, 2))))
    assert_less(fnorm / datasize, datasize)

