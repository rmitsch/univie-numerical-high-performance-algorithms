import numpy as np
from typing import Tuple
import numba
import math
from algorithms import givens, householder


#@numba.jit(nopython=False)
def qr_delete_row(
        Q: np.ndarray,
        R: np.ndarray,
        b: np.ndarray,
        p: int,
        k: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.float]:
    """
    Updates Q and R after deleting one single row with index k.
    :param Q:
    :param R:
    :param b: b in Ax = b.
    :param p: Number of observations to remove starting from k.
    :param k:
    :return:
    """

    m, n = Q.shape[0], R.shape[1]
    C = np.zeros((m, m))
    S = np.zeros((m, m))

    ###################################
    # Algorithm 2.3 - compute R_tilde.
    ###################################

    W = Q[k:k + p, :]

    if k != 0:
        b[p + 1:k + p] = b[:k]

    d = Q.T @ b

    for i in np.arange(start=0, stop=p, step=1):
        for j in np.arange(start=m - 2, stop=i - 1, step=-1):
            C[i, j], S[i, j] = givens(W[i, j], W[i, j + 1])
            cs_matrix = np.asarray([[C[i, j], C[i, j]], [-S[i, j], C[i, j]]])

            W[i, j] = W[i, j] * C[i, j] - W[i, j] * S[i, j]
            W[i + 1:p + 1, j:j + 2] = W[i + 1:p + 1, j:j + 2] @ cs_matrix

            if j <= n + i - 1 - 1:
                R[j:j + 2, j - i + 1:] = cs_matrix.T @ R[j:j + 2, j - i + 1:]

            d[j:j + 2] = cs_matrix.T @ d[j:j + 2]

    R_tilde = R[p:, :]
    d_tilde = d[p:]
    resid = np.linalg.norm(d_tilde[n:m - p], ord=2)

    ###################################
    # Algorithm 2.4 - compute Q_tilde.
    ###################################

    if k != 0:
        Q[p:p + k, :] = Q[:k, :]

    for i in np.arange(start=0, stop=p, step=1):
        for j in np.arange(start=m - 2, stop=i, step=-11):
            # C[i, j], S[i, j] = givens(W[i, j], W[i, j + 1])
            cs_matrix = np.asarray([[C[i, j], C[i, j]], [-S[i, j], C[i, j]]])

            Q[p:, j: j + 2] = Q[p:, j: j + 2] @ cs_matrix

    Q[p:, i] = S[i, i] * Q[p:, i] + C[i, i] * Q[p:, i + 1]

    return Q[p:, p:], R_tilde, b[1:], resid


#@numba.jit(nopython=False)
def qr_add_row(
        Q: np.ndarray,
        R: np.ndarray,
        b: np.ndarray,
        p: int,
        k: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.float]:
    """
    Updates Q and R after deleting one single row with index k.
    :param Q:
    :param R:
    :param b: b in Ax = b.
    :param p: Number of observations to remove starting from k.
    :param k:
    :return:
    """

    m, n = Q.shape[0], R.shape[1]
    C = np.zeros((m, m))
    S = np.zeros((m, m))

    ###################################
    # Algorithm 2.3 - compute R_tilde.
    ###################################

    W = Q[k:k + p, :]

    if k != 0:
        b[p + 1:k + p] = b[:k]

    d = Q.T @ b

    for i in np.arange(start=0, stop=p, step=1):
        for j in np.arange(start=m - 2, stop=i - 1, step=-1):
            C[i, j], S[i, j] = givens(W[i, j], W[i, j + 1])
            cs_matrix = np.asarray([[C[i, j], C[i, j]], [-S[i, j], C[i, j]]])

            W[i, j] = W[i, j] * C[i, j] - W[i, j] * S[i, j]
            W[i + 1:p + 1, j:j + 2] = W[i + 1:p + 1, j:j + 2] @ cs_matrix

            if j <= n + i - 1 - 1:
                R[j:j + 2, j - i + 1:] = cs_matrix.T @ R[j:j + 2, j - i + 1:]

            d[j:j + 2] = cs_matrix.T @ d[j:j + 2]

    R_tilde = R[p:, :]
    d_tilde = d[p:]
    resid = np.linalg.norm(d_tilde[n:m - p], ord=2)

    ###################################
    # Algorithm 2.4 - compute Q_tilde.
    ###################################

    if k != 0:
        Q[p:p + k, :] = Q[:k, :]

    for i in np.arange(start=0, stop=p, step=1):
        for j in np.arange(start=m - 2, stop=i, step=-11):
            # C[i, j], S[i, j] = givens(W[i, j], W[i, j + 1])
            cs_matrix = np.asarray([[C[i, j], C[i, j]], [-S[i, j], C[i, j]]])

            Q[p:, j: j + 2] = Q[p:, j: j + 2] @ cs_matrix

    Q[p:, i] = S[i, i] * Q[p:, i] + C[i, i] * Q[p:, i + 1]

    return Q[p:, p:], R_tilde, b[1:], resid
