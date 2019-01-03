import numpy as np
from typing import Tuple
from algorithms.l1 import givens


def householder(alpha: float, x: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Computes Householder transformation as specified in supplied paper.
    :param alpha:
    :param x:
    :return: v, tau.
    """

    s = np.power(np.linalg.norm(x, ord=2), 2)
    v = x

    if s == 0:
        tau = 0
    else:
        t = np.sqrt(alpha * alpha + s)

        if alpha <= 0:
            v_one = alpha - t
        else:
            v_one = -s / (alpha + 1)

        tau = 2 * np.power(v_one, 2) / (s + np.power(v_one, 2))
        v = v / v_one

    return v, tau


#@numba.jit(nopython=False)
def qr_delete_rows(
        Q: np.ndarray,
        R: np.ndarray,
        b: np.ndarray,
        p: int,
        k: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.float]:
    """
    Updates Q and R after deleting a set of rows starting with index k.
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
        b[p + 1:k + p] = b[:k - 1]

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

    return Q[p:, p:], R_tilde, b[p:], resid


#@numba.jit(nopython=False)
def qr_add_rows(
        Q: np.ndarray,
        R: np.ndarray,
        b: np.ndarray,
        p: int,
        U: np.ndarray,
        e: np.ndarray,
        k: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.float]:
    """
    Updates Q and R after adding a set of rows starting with index k.
    :param Q:
    :param R:
    :param b: b in Ax = b.
    :param p: Number of observations to remove starting from k.
    :param U: Rows to insert.
    :param e: Ux = e. To be inserted in b at index k. Equivalent to mu in single row addition.
    :param k:
    :return:
    """

    m, n = Q.shape[0], R.shape[1]
    V = np.zeros((m, m))
    tau = np.zeros((m,))

    ###################################
    # Algorithm 2.8 - compute R_tilde.
    ###################################

    d = Q.T @ b
    R_j = None
    for j in np.arange(start=0, stop=n, step=1):
        V[:p, j], tau[j] = householder(R[j, j], U[:p, j])

        R_j = R[j, j + 1:]
        R[j, j:] = \
            (1 - tau[j]) * R[j, j:] - \
            tau[j] * V[:p, j].T @ U[:p, j:]

        if j < n - 1:
            U[:p, j + 1:] = U[:p, j + 1:] - \
                tau[j] * np.outer(V[:p, j], R_j) - \
                tau[j] * np.outer(V[:p, j], V[:p, j].T @ U[:p, j + 1:])

        d_j = d[j]
        d[j] = \
            (1 - tau[j]) * d[j] - \
            tau[j] * (V[:p, j].T @ e[:p])

        e[:p] = e[:p] - \
            tau[j] * np.outer(V[:p, j], d_j) - \
            tau[j] * np.outer(V[:p, j], V[:p, j].T @ e[:p])

    R_tilde = np.zeros((m + p, n))
    R_tilde[:m] = R
    d_tilde = np.zeros((m + p,))
    d_tilde[:m] = d[:, 0]
    d_tilde[m:] = e[:, 0]
    resid = np.linalg.norm(d_tilde[n:m + p], ord=2)

    ###################################
    # Algorithm 2.9 - compute Q_tilde.
    ###################################

    Q_tilde = np.zeros((m + p, m + p))
    Q_tilde[:m, :m] = Q
    Q_tilde[m:, m:] = np.eye(p, p)

    if k != m:
        Q_tilde[:k - 1] = Q_tilde[:k - 1, :]
        Q_tilde[k - 1:k - 1 + p] = Q_tilde[m:m + p, :]
        Q_tilde[k - 1 + p:] = Q_tilde[k:m, :]

    for j in np.arange(start=0, stop=n, step=1):
        Q_tilde_k = Q_tilde[:, j]
        Q_tilde[:, j] = \
            Q_tilde[:, j] * (1 - tau[j]) - \
            Q_tilde[:, m:] @ (tau[j] * V[:p, j])
        Q_tilde[:, m:] = Q_tilde[:, m:] - \
            tau[j] * np.outer(Q_tilde_k, V[:p, j].T) - \
            tau[j] * np.outer(Q_tilde[:, m:] @ V[:p, j], V[:p, j].T)

    b_tilde = np.zeros((m + p, 1))
    b_tilde[:k] = b[:k]
    b_tilde[k:k + p] = e
    b_tilde[k + p:] = b[k:]

    return Q_tilde, R_tilde, b_tilde, resid


#@numba.jit(nopython=False)
def qr_delete_cols(
        Q: np.ndarray,
        R: np.ndarray,
        b: np.ndarray,
        k: int,
        p: int
) -> Tuple[np.ndarray, np.ndarray, np.float]:
    """
    Updates Q and R by deleting a set of columns starting with index k.
    :param Q:
    :param R:
    :param b:
    :param k:
    :param p: Number of cols to delete.
    :return:
    """

    m, n = Q.shape[0], R.shape[1]
    V = np.zeros((m, m))
    tau = np.zeros((m,))

    ###################################
    # Algorithm 2.15 - compute R_tilde.
    ###################################

    d_tilde = Q.T @ b
    R[:, k:n - p] = R[:, k + p:n]

    for j in np.arange(start=k, stop=n - p, step=1):
        V[:p, j], tau[j] = householder(R[j, j], R[j + 1:j + p + 1, j])
        R[j, j] = R[j, j] - \
            tau[j] * R[j, j] - \
            tau[j] * (V[:p, j].T @ R[j + 1:j + p + 1, j])

        v = np.ones((p + 1, 1))
        v[1:] = V[:p, j]

        if j < n - p - 1:
            R[j:j + p + 1, j + 1:n - p + 1] = R[j:j + p + 1, j + 1:n - p + 1] - \
                tau[j] * v @ (v.T @ R[j:j + p + 1, j + 1:n - p + 1])

        # Note d_tilde[j:j + p, j + 1] is supposed to be first term, but indexing as specified is not possible.
        d_tilde[j:j + p + 1] = d_tilde[j:j + p + 1] - \
            tau[j] * v @ (v.T @ d_tilde[j:j + p + 1])

    R_tilde = np.triu(R[:, :n - p])
    resid = np.linalg.norm(d_tilde[n:m], ord=2)

    ###################################
    # Algorithm 2.16 - compute Q_tilde.
    ###################################

    for j in np.arange(start=k, stop=n - p, step=1):
        v = np.ones((p + 1, 1))
        v[1:] = V[:p, j]

        Q[:, j:j + p + 1] = Q[:, j:j + p + 1] - \
            tau[j] * (Q[:, j:j + p + 1] @ v) @ v.T

    return Q, R_tilde, resid


#@numba.jit(nopython=False)
def qr_add_cols(
        Q: np.ndarray,
        R: np.ndarray,
        b: np.ndarray,
        p: int,
        U: np.ndarray,
        k: int
) -> Tuple[np.ndarray, np.ndarray, np.float]:
    """
    Updates Q and R after adding a set of columns starting with index k.
    :param Q:
    :param R:
    :param b: b in Ax = b.
    :param p: Number of observations to remove starting from k.
    :param U: Rows to insert.
    :param e: Ux = e. To be inserted in b at index k. Equivalent to mu in single row addition.
    :param k:
    :return:
    """

    m, n = Q.shape[0], R.shape[1]
    C = np.zeros((m, m))
    S = np.zeros((m, m))

    ###################################
    # Algorithm 2.21 - compute R_tilde.
    ###################################

    U = Q.T @ U
    d_tilde = Q.T @ b

    for j in np.arange(start=0, stop=p, step=1):
        for i in np.arange(start=m - 1, stop=k + j - 1):
            C[i, j], S[i, j] = givens(U[i - 1, j], U[i, j])
            cs_matrix = np.asarray([[C[i, j], C[i, j]], [-S[i, j], C[i, j]]])

            U[i - 1, j] = C[i, j] * U[i - 1, j] - S[i, j] * U[i, j]

            if j + 1 < p:
                U[i - 1:i + 1, j + 1:p] = cs_matrix.T @ U[i - 1:i + 1, j + 1:p]

            if i <= n + j:
                R[i - 1:i + 1, i - j:] = cs_matrix.T @ d_tilde[i - 1:i + 1]

            d_tilde[i - 1:i + 1] = cs_matrix.T @ d_tilde[i - 1:i + 1]

    R_tilde = np.zeros((m, p + n))
    if k == 0:
        R_tilde[:, :p] = U
        R_tilde[:, p:] = R
    elif k == n:
        R_tilde[:, :n] = R
        R_tilde[:, n:] = U
    else:
        R_tilde[:, :k - 1] = R[:, :k - 1]
        R_tilde[:, k - 1:k - 1 + p] = U
        R_tilde[:, k - 1 + p:] = R[:, k - 1:]
    R_tilde = np.triu(R_tilde)

    resid = np.linalg.norm(d_tilde[n:], ord=2)

    ###################################
    # Algorithm 2.22 - compute Q_tilde.
    ###################################

    for j in np.arange(start=0, stop=p, step=1):
        for i in np.arange(start=m - 1, stop=k + j - 1):
            cs_matrix = np.asarray([[C[i, j], C[i, j]], [-S[i, j], C[i, j]]])
            Q[:, i - 1:i + 1] = Q[:, i - 1:i + 1] @ cs_matrix

    return Q, R_tilde, resid
