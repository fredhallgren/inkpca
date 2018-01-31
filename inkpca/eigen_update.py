
# -*- coding: utf-8 -*-

import numpy as np
from numpy import dot, zeros
from scipy import optimize
from matplotlib import pyplot as plt


def expand_eigensystem(L, U, e):
    """
    Expand eigensystem with new eigenpair orthogonal to the existing ones

    Parameters
    ----------
    L : numpy.ndarray, 1d
        Current eigenvalues
    U : numpy.ndarray, 2d
        Current eigenvectors
    e : float
        New eigenvalue
    """
    m = L.shape[0]
    L = np.r_[L, [e]]
    U = np.r_[np.c_[U, zeros((m,1))], zeros((1,m+1))]
    U[m,m] = 1

    return L, U

def update_eigensystem(L, U, v, sigma):
    """
    Perform rank one update to eigensystem. Requires the eigenpairs to be
    ordered

    Parameters
    ----------
    L : numpy.ndarray, 1d
        Current eigenvalues
    U : numpy.ndarray, 2d
        Current eigenvectors
    v : numpy.ndarray, 2d
        Column vector for rank one update
    sigma : float
        update coefficient

    """
    n = len(L)
    z = dot(U.T, v)

    L_tilde = rank_one_eigenvalues(L, z, sigma)
    if not isinstance(L_tilde, np.ndarray):
        return None, None

    U_tilde = rank_one_eigenvectors(L, L_tilde, U, z)

    return L_tilde, U_tilde

def rank_one_eigenvalues(L, z, sigma, eps=1E-12):
    """
    Rank one update of eigenvalues of diagonal matrix

    """
    n = len(L)
    z2 = np.power(z, 2)
    z2 = z2[:,0]
    factor = sigma * dot(z.T, z)[0]

    if sigma > 0:
        bounds = np.r_[L, L[-1]+factor]
    elif sigma < 0:
        bounds = np.r_[L[0]+factor, L]

    L_tilde = zeros(n)
    for i in np.arange(n):
        a = bounds[i] + eps
        b = bounds[i+1] - eps
        if omega(a,sigma,z2,L) * omega(b,sigma,z2,L) > 0:
            L_tilde = None
            break
        else:
            L_tilde[i] = optimize.brentq(lambda x: omega(x, sigma, z2, L), a, b)

    return L_tilde

def omega(x, sigma, z2, L):
    """
    Objective function

    """
    return 1 + sigma * np.sum(z2 / (L - x))

def rank_one_eigenvectors(L, L_tilde, U, z):
    """
    Adjust eigenvectors

    """
    n = len(L)
    LL = zeros((n,n))
    LL[:,:] = np.expand_dims(L,1)
    Ldiff = LL - L_tilde
    LL = np.power(Ldiff, -1)
    Dz = LL*z
    norm = np.sqrt(np.sum(np.power(Dz,2),0)) # row
    U_tilde = dot(U, Dz) / norm

    return U_tilde

