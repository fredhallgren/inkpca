
# -*- coding: utf-8 -*-

from __future__ import division, print_function

from kernels import kernel_matrix, adjust_K
from eigen_update import expand_eigensystem, update_eigensystem

from copy import copy, deepcopy

import numpy as np
from numpy import dot, diag, ones
from scipy import linalg


def kernel_error(*args, **kwargs):
    raise ValueError("Kernel function not specified!")


class IncrKPCA(object):
    """
    Incremental kernel PCA

    Parameters
    ----------
    X : numpy.ndarray, 2d
        Data matrix
    m0 : int
        Initial size of kernel matrix (or Nyström subset)
    mmax : int
        Maximum size of kernel matrix (or Nyström subset)
    kernel : callable
        Kernel function
    adjust : bool
        Whether to adjust the mean
    nystrom : bool
        Calculate incremental Nyström approximation instead
    maxiter : int
        Maximum number of iterations

    Yields
    ------
    The iteration number, the calculated eigenvectors and eigenvalues, and in
    the case of the Nyström method, also the approximate eigenvalues and
    eigenvectors

    """

    def __init__(self, X, m0, mmax=None, kernel=kernel_error, adjust=False,
                     nystrom=False, maxiter=500):

        # Setup default arguments
        n = X.shape[0]
        self.X = X
        self.i = m0
        self.m0 = m0
        self.j = 0
        self.n = n
        self.maxiter = maxiter

        if mmax is None:
            mmax = n
        self.mmax = min(mmax, n)

        self.idx = np.random.permutation(n)
        self.kernel = kernel
        self.adjust = adjust
        self.nystrom = nystrom

        # Initial eigensystem
        cols = self.idx[:m0]
        K_mm = kernel_matrix(X, kernel, cols, cols)

        if self.adjust:
            self.L, self.U, self.capsig, self.K1 = init_vars(K_mm)
        else:
            self.L, self.U = linalg.eigh(K_mm)

        if self.nystrom:
            self.K_nm = kernel_matrix(X, kernel, range(n), cols)

    def next(self):
        """
        Initiate the next iteration

        """
        if self.i == self.mmax:
            raise StopIteration
        if self.j == len(self.idx)-self.m0 or self.j == self.maxiter:
            raise StopIteration

        if not self.adjust:
            rc = self.update_eig()
        else:
            rc = self.update_eig_adjust()

        out = (self.i-1, self.L, self.U)
        if self.nystrom:
            if not rc:
                self.L_nys, self.U_nys = nystrom_approximation(
                        self.L, self.U, self.K_nm)
                out = (self.i-1, self.L, self.U, self.L_nys, self.U_nys)

        self.j += 1

        if rc:
            return self.next()
        else:
            return out

    def update_K_nm(self):
        """
        Update K_nm for one iteration by adding another column

        """
        i = self.i
        K_ni = kernel_matrix(self.X, self.kernel, range(self.n), [self.idx[i]])
        self.K_nm = np.c_[self.K_nm, K_ni]

    def update_eig(self):
        """
        Update the eigendecomposition of K with an additional data point

        """
        i = self.i # index of new data points / size of existing K
        col = self.idx[i]
        cols = self.idx[:i+1]
        sigma, k1, k0 = create_update_terms(self.X, cols, col, self.kernel)

        L, U = expand_eigensystem(self.L, self.U, k0[-1][0])

        # Order the eigenpairs and update terms
        idx = np.argsort(L)
        L = L[idx]
        U = U[:,idx]
        U = U[idx,:]
        k1 = k1[idx,:]
        k0 = k0[idx,:]

        L, U = update_eigensystem(L, U, k1, sigma)
        if isinstance(L, np.ndarray):
            L, U = update_eigensystem(L, U, k0, -sigma)

        if isinstance(L, np.ndarray):
            if self.nystrom:
                self.update_K_nm()
            self.idx[:i+1] = self.idx[:i+1][idx] # reorder index
            if self.nystrom:
                self.K_nm = self.K_nm[:,idx] # Reorder columns
            self.i, self.L, self.U = i+1, L, U
            rc = 0
        else: # Ignore data example
            self.idx[i:-1] = self.idx[i+1:]
            self.idx = self.idx[:-1]
            rc = 1

        return rc

    def update_eig_adjust(self):
        """
        Update the kernel PCA solution including adjustment of the mean.

        """
        i = self.i
        col = self.idx[i]
        cols = self.idx[:i+1]
        k = kernel_matrix(self.X, self.kernel, cols, [col]) # OK
        a = k[:-1,:]
        a_sum = np.sum(a)
        k_sum = np.sum(k)
        capsig2 = self.capsig + 2 * a_sum + k[-1,0]
        C = -self.capsig/i**2 + capsig2/(i+1)**2
        u =  self.K1/(i*(i+1)) - a/(i+1) + 0.5 * C * ones((i,1))
        u1 = 1 + u
        u2 = 1 - u
        sigma_u = 0.5

        K1 = np.r_[self.K1 + a, [[k_sum]]]
        capsig = capsig2
        v = k - (ones((i+1,1)) * k_sum + K1 - capsig/(i+1)) / (i+1)
        v1 = deepcopy(v)
        v2 = deepcopy(v)
        v0 = copy(v[-1,0])
        v1[-1,0] = v0 / 2
        v2[-1,0] = v0 / 4
        sigma_k = 4 / v0

        # Apply rank one updates
        L, U = update_eigensystem(self.L, self.U, u1, sigma_u)
        if isinstance(L, np.ndarray):
            L, U = update_eigensystem(L, U, u2, -sigma_u)
        if isinstance(L, np.ndarray):
            L, U = expand_eigensystem(L, U, v0/4)

            # Ordering
            idx = np.argsort(L)
            L = L[idx]
            U = U[:,idx]
            U = U[idx,:]
            v1 = v1[idx,:]
            v2 = v2[idx,:]

            L, U = update_eigensystem(L, U, v1, sigma_k)

        if isinstance(L, np.ndarray):
            L, U = update_eigensystem(L, U, v2, -sigma_k)

        if isinstance(L, np.ndarray):
            #f self.nystrom:
            #    self.update_K_nm()
            K1 = K1[idx,:]
            self.idx[:i+1] = self.idx[:i+1][idx] # Reorder index
            if self.nystrom:
                self.K_nm = self.K_nm[:,idx] # Reorder columns
            self.i, self.L, self.U, self.K1 = i+1, L, U, K1
            self.capsig = capsig
            rc = 0
        else: # Ignore data example
            self.idx[i:-1] = self.idx[i+1:]
            self.idx = self.idx[:-1]
            rc = 1

        return rc

    def get_idx_array(self):
        return self.idx[:self.mmax]

    def __iter__(self):
        return self

def init_vars(K_mm):
    """
    Create initial eigenpairs and adjustment variables

    Parameters
    ----------
    K_mm : np.ndarray, 2d
        Initial kernel matrix

    Returns
    -------
    Initial eigenvalues L and eigenvectors U, sum of all values of K_mm
    (capsig), sum of the rows of K_mm (K1)

    """
    m0 = K_mm.shape[0]
    Kp = adjust_K(K_mm)
    L, U = linalg.eigh(Kp)
    capsig = np.sum(np.sum(K_mm))
    K1 = dot(K_mm, ones((m0, 1)))

    return L, U, capsig, K1

def create_update_terms(X, cols, col, kernel):
    """
    Create the terms supplied to eigenvalue update algorithm

    Parameters
    ----------
    X : np.ndarray, 2d
        Data matrix
    cols : np.ndarray, 1d
        Indices of columns to create the kernel matrix
    col : float
        The additional column index

    Returns
    -------
    Parameters supplied to update algorithm for
    eigendecomposition

    """
    k1 = kernel_matrix(X, kernel, cols, [col])
    k = copy(k1[-1][0])
    k1[-1] = k / 2
    k0 = deepcopy(k1) # numpy pass by reference
    k0[-1] = k / 4
    sigma = 4 / k

    return sigma, k1, k0

def nystrom_approximation(L, U, K_nm):
    """
    Create the Nyström approximations to the eigenpairs of the kernel matrix

    Parameters
    ----------
    L : numpy.ndarray, 2d
        eigenvector matrix for the matrix K_mm
    U : numpy.ndarray, 1d
        eigenvalues for the matrix K_mm
    K_nm : numpy.ndarray, 2d
        the m sampled columns of K

    Returns
    -------
    Nyström approximate eigenvalues L and eigenvectors U

    """
    n, m = K_nm.shape
    L_nys = n/m * L
    U_nys = np.sqrt(m/n) * dot(K_nm, dot(U, diag(1/L)))

    return L_nys, U_nys

