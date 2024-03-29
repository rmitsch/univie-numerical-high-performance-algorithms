import numpy as np
from typing import Tuple
import numba
import math


@numba.jit(nopython=True)
def householder_vec(x: np.ndarray) -> Tuple[np.ndarray, int]:
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


@numba.jit(nopython=True)
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


@numba.jit(nopython=True)
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
        v, tau = householder_vec(R[j:, j])
        H = np.identity(m)
        H[j:, j:] -= tau * v.reshape(-1, 1) * v
        R = H @ R
        Q = H @ Q

    return Q.T, R  # Q[:n].T, np.triu(R[:n])


@numba.jit(nopython=True)
def compute_residual(A_prime: np.ndarray, A: np.ndarray):
    """
    Compute residual norm for approximation A_prime of A.
    :param A_prime: Approximation of A.
    :param A: Original matrix.
    :return:
    """
    return np.linalg.norm(A_prime - A) / np.linalg.norm(A)


@numba.jit(nopython=True)
def matmul(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Numba-assisted matrix multiplication with numpy.
    :param A:
    :param B:
    :return:
    """
    return A @ B


# @numba.jit(nopython=False)
def qr_delete_row(
        Q: np.ndarray,
        R: np.ndarray,
        b: np.ndarray,
        k: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.float]:
    """
    Updates Q and R after deleting one single row with index k.
    :param Q:
    :param R:
    :param b: b in Ax = b.
    :param k:
    :return:
    """

    m, n = Q.shape[0], R.shape[1]
    c = np.zeros((m,))
    s = np.zeros((m,))

    ###################################
    # Algorithm 2.1 - compute R_tilde.
    ###################################

    q = Q[k, :].T

    if k != 0:
        b[1:k + 1] = b[:k]

    d = Q.T @ b

    for j in np.arange(start=m - 2, step=-1, stop=-1):
        c[j], s[j] = givens(q[j], q[j + 1])
        cs_matrix = np.asarray([[c[j], s[j]], [-s[j], c[j]]])
        q[j] = c[j] * q[j] - s[j] * q[j + 1]

        if j <= n - 1:
            R[j:j + 2, j:] = cs_matrix.T @ R[j:j + 2, j:]

        d[j:j + 2] = cs_matrix.T @ d[j:j + 2]

    R_tilde = R[1:, :]
    d_tilde = d[1:]
    resid = np.linalg.norm(d_tilde[n:m - 1], ord=2)

    ###################################
    # Algorithm 2.2 - compute Q_tilde.
    ###################################

    if k != 0:
        Q[1:k + 1, :] = Q[:k, :]

    for j in np.arange(start=m - 2, step=-1, stop=0):
        cs_matrix = np.asarray([[c[j], s[j]], [-s[j], c[j]]])
        Q[1:, j:j + 2] = Q[1:m, j:j + 2] @ cs_matrix

    Q[1:, 1] = s[0] * Q[1:, 0] + c[0] * Q[1:, 1]

    return Q[1:, 1:], R_tilde, b[1:], resid


@numba.jit(nopython=False)
def qr_add_row(
        Q: np.ndarray,
        R: np.ndarray,
        u: np.ndarray,
        mu: float,
        b: np.ndarray,
        k: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.float]:
    """
    Updates Q and R by adding one single row with index k.
    :param Q:
    :param R:
    :param u: Vector to insert in A.
    :param mu: Value to insert in b.
    :param b: b in Ax = b.
    :param k:
    :return:
    """

    m, n = R.shape
    c = np.zeros((m,))
    s = np.zeros((m,))

    b_tilde = np.zeros((m + 1, 1))
    b_tilde[:k] = b[:k]
    b_tilde[k] = mu
    b_tilde[k + 1:] = b[k:]

    ###################################
    # Algorithm 2.6 - compute R_tilde.
    ###################################

    d = Q.T @ b
    for j in np.arange(0, n):
        c[j], s[j] = givens(R[j, j], u[j])
        R[j, j] = c[j] * R[j, j] - s[j] * u[j]

        t1 = R[j, j + 1:]
        t2 = u[j + 1:]
        R[j, j + 1:] = c[j] * t1 - s[j] * t2
        u[j + 1:] = s[j] * t1 + c[j] * t2

        t1 = d[j]
        t2 = mu
        d[j] = c[j] * t1 - s[j] * t2
        mu = s[j] * t1 + c[j] * t2

    R_tilde = np.append(R, np.zeros((1, n)), axis=0)
    d_tilde = np.append(d, mu)
    resid = np.linalg.norm(d_tilde[n:m + 1], ord=2)

    ###################################
    # Algorithm 2.7 - compute Q_tilde.
    ###################################

    Q_tilde = np.zeros((m + 1, m + 1))
    Q_tilde[:m, :m] = Q
    Q_tilde[m, m] = 1

    if k != m:
        # Note that paper refers to Q here, not Q~. Specified indexing is not possible with Q though.
        Q_m = Q_tilde[m]
        Q_tilde[k + 1:] = Q_tilde[k:m]
        Q_tilde[k] = Q_m

    for j in np.arange(start=0, stop=n, step=1):
        t1 = Q_tilde[:, j]
        t2 = Q_tilde[:, m]
        Q_tilde[:, j] = c[j] * t1 - s[j] * t2
        Q_tilde[:, m] = c[j] * t1 - s[j] * t2

    return Q_tilde, R_tilde, b_tilde, resid


def qr_insert_col():
    pass


@numba.jit(nopython=False)
def qr_delete_col(
        Q: np.ndarray,
        R: np.ndarray,
        b: np.ndarray,
        k: int
) -> Tuple[np.ndarray, np.ndarray, np.float]:
    """
    Updates Q and R by deleting one single column at index k.
    :param Q:
    :param R:
    :param b:
    :param k:
    :return:
    """

    m, n = Q.shape[0], R.shape[1]
    c = np.zeros((m,))
    s = np.zeros((m,))

    ###################################
    # Algorithm 2.13 - compute R_tilde.
    ###################################

    d_tilde = Q.T @ b
    R[:, k:n - 1] = R[:, k + 1:]

    for j in np.arange(start=k, stop=n, step=1):
        c[j], s[j] = givens(R[j, j], R[j + 1, j])
        cs_matrix = np.asarray([[c[j], s[j]], [-s[j], c[j]]])

        R[j, j] = c[j] * R[j, j] - s[j] * R[j + 1, j]
        R[j:j + 2, j + 1:n - 1] = cs_matrix.T @ R[j:j + 2, j + 1:n - 1]

        d_tilde[j:j + 2] = cs_matrix.T @ d_tilde[j:j + 2]

    R_tilde = np.triu(R[:, :n - 1])
    resid = np.linalg.norm(d_tilde[n:m], ord=2)

    ###################################
    # Algorithm 2.14 - compute Q_tilde.
    ###################################

    for j in np.arange(start=k, stop=n, step=1):
        cs_matrix = np.asarray([[c[j], s[j]], [-s[j], c[j]]])
        Q[:, j:j + 2] = Q[:, j:j + 2] @  cs_matrix

    return Q, R_tilde, resid


@numba.jit(nopython=False)
def qr_add_col(
        Q: np.ndarray,
        R: np.ndarray,
        u: np.ndarray,
        b: np.ndarray,
        k: int
) -> Tuple[np.ndarray, np.ndarray, np.float]:
    """
    Updates Q and R by adding one single column with index k.
    :param Q:
    :param R:
    :param u: Vector to insert in A.
    :param b: b in Ax = b.
    :param k:
    :return:
    """

    m, n = Q.shape[0], R.shape[1]
    c = np.zeros((m,))
    s = np.zeros((m,))

    ###################################
    # Algorithm 2.19 - compute R_tilde.
    ###################################

    u = Q.T @ u
    d_tilde = Q.T @ b

    for i in np.arange(start=m - 1, stop=k, step=-1):
        c[i], s[i] = givens(u[i - 1], u[i])
        cs_matrix = np.asarray([[c[i], s[i]], [-s[i], c[i]]])
        u[i - 1] = c[i] * u[i - 1] - s[i] * u[i]  # R_tilde[i} - wtf?

        if i <= n:
            R[i - 1:i + 1, i - 1:] = cs_matrix.T @ R[i - 1:i + 1, i - 1:]

    R_tilde = np.zeros((m, n + 1))
    R_tilde[:, k] = u

    if k == 0:
        R_tilde[:, k + 1:] = R

    elif k == n:
        R_tilde[:, :k] = R

    else:
        R_tilde[:, :k] = R[:, :k]
        R_tilde[:, k + 1:] = R[:, k:]

    R_tilde = np.triu(R_tilde)
    resid = np.linalg.norm(d_tilde[n:m], ord=2)

    ###################################
    # Algorithm 2.20 - compute Q_tilde.
    ###################################

    for i in np.arange(start=m-1, stop=k, step=-1):
        cs_matrix = np.asarray([[c[i], s[i]], [-s[i], c[i]]])
        Q[:, i - 1:i + 1] = Q[:, i - 1:i + 1] @ cs_matrix

    return Q, R_tilde, resid
