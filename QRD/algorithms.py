import numpy as np
from typing import Tuple
import logging
import numba
import math


@numba.jit(nopython=True)
def householder(x: np.ndarray) -> Tuple[np.ndarray, int]:
    """
    Computes v and tau for Householder transformation of x.
    Note that
    https://stackoverflow.com/questions/53489237/qr-decomposition-with-householder-transformations-introduction-of-zeroes-uncle
    helped in getting the implementation right.
    :param x: Vector.
    :return:
    """

    # Use vectorized approach as shown in https://rosettacode.org/wiki/QR_decomposition#Python.
    v = x / (x[0] + np.copysign(np.linalg.norm(x), x[0]))
    v[0] = 1
    tau = 2 / (v.T @ v)

    # Down here: Approach as in paper with slight modifications.
    # alpha = x[0]
    # s = np.power(np.linalg.norm(x[1:]), 2)
    # v = x.copy()
    #
    # if s == 0:
    #     tau = 0
    # else:
    #     t = np.sqrt(alpha**2 + s)
    #     v[0] = alpha - t if alpha <= 0 else -s / (alpha + t)
    #
    #     tau = 2 * v[0]**2 / (s + v[0]**2)
    #     v /= v[0]

    return v, tau


@numba.jit(nopython=False)
def givens(a: float, b: float) -> Tuple[float, float]:
    """
    Calculate Givens rotation with two scalars.
    :param a:
    :param b:
    :return:
    """

    if b == 0:
        c = 1
        s = 0
    else:
        if abs(b) >= abs(a):
            t = -a / b
            s = 1 / math.sqrt(1 + t * t)
            c = s * t
        else:
            t = -b / a
            c = 1 / math.sqrt(1 + t * t)
            s = c * t

    return c, s


@numba.jit(nopython=False)
def qr_decomposition(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Decomposes rectangular matrix A in matrices Q and R.
    Note that
    https://stackoverflow.com/questions/53489237/qr-decomposition-with-householder-transformations-introduction-of-zeroes-uncle
    helped in getting the implementation right.
    :param A:
    :return:
    """

    m, n = A.shape
    R = A.copy()
    Q = np.identity(m)

    for j in range(0, n):
        # Apply Householder transformation.
        v, tau = householder(R[j:, j])
        H = np.identity(m)
        H[j:, j:] -= tau * v.reshape(-1, 1) * v
        R = H @ R
        Q = H @ Q

    # triu is used here mainly for cosmetic reasons - we discard of the entries under the diagonale.
    return Q[:n].T, np.triu(R[:n])


@numba.jit(nopython=False)
def compute_residual(A_prime: np.ndarray, A: np.ndarray):
    """
    Compute residual norm for approximation A_prime of A.
    :param A_prime: Approximation of A.
    :param A: Original matrix.
    :return:
    """
    return np.linalg.norm(A_prime - A) / np.linalg.norm(A)


def compile_functions_with_numba():
    """
    Calls all numba-decorated function with dummy data so that they are compiled lazily.
    :return:
    """

    A = np.random.rand(5, 5)
    A_prime = qr_decomposition(A)
    matmul(A, A_prime)


@numba.jit(nopython=False)
def matmul(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Numba-assisted matrix multiplication with numpy.
    :param A:
    :param B:
    :return:
    """
    return A @ B


def qr_insert_row():
    pass


@numba.jit(nopython=False)
def qr_delete_row(Q: np.ndarray, R: np.ndarray, b: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Updates Q and R after deleting one single row with index k.
    :param Q:
    :param R:
    :param b: b in Ax = b.
    :param k:
    :return:
    """

    m, n = Q.shape

    q_t = Q[k, 0:m + 1]
    q = q_t.T

    if k != 1:
        b[2:k + 1] = b[1:k]

    d = np.dot(Q.T * b)

    for j in np.arange(start=m - 1, step=-1, stop=1):
        c, s = givens(q[j], q[j + 1])
        # todo cont. here
    return Q, R


def qr_insert_col():
    pass


def qr_delete_col():
    pass

