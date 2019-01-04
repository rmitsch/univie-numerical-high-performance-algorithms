import numpy as np
from typing import Tuple
from algorithms.l1 import givens, qr_decomposition
from algorithms.l2 import householder


#@numba.jit(nopython=False)
def qr_add_rows(
        Q: np.ndarray,
        R: np.ndarray,
        b: np.ndarray,
        U: np.ndarray,
        e: np.ndarray,
        k: int,
        bs: int = 10,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.float]:
    """
    Updates Q and R after adding a set of rows starting with index k.
    :param Q:
    :param R:
    :param b: b in Ax = b.
    :param U: Rows to insert.
    :param e: Ux = e. To be inserted in b at index k. Equivalent to mu in single row addition.
    :param bs: Block size.
    :param k:
    :return:
    """

    m, n = Q.shape[0], R.shape[1]
    p = U.shape[0]
    V = np.zeros((p, n))
    tau = np.zeros((n,))

    ###################################
    # Algorithm 2.10 - compute R_tilde.
    ###################################

    # Compute V.
    for j in np.arange(start=0, stop=n, step=1):
        V[:p, j], tau[j] = householder(R[j, j], U[:p, j])

    d = Q.T @ b
    for k in np.arange(start=0, stop=n, step=bs):
        jb = min(bs, n - k + 1)

        if k + jb <= n:
            T = np.zeros((jb, jb))
            for j in np.arange(start=k, stop=k + jb - 2, step=1):
                if j == k:
                    T[0, 0] = tau[j]
                else:
                    T[:j - k, j - k + 2] = - (
                            tau[j] * T[:j - k, j - k + 2] *
                            (V[:, k:j - 0].T @ V[:, j])
                    )
                    T[j - k + 1, j - k + 1] = tau[j]

            T_v = T.T @ V[:, k:k + jb].T
            T_e = T_v @ e
            T_u = T_v @ U[:, k + jb:]

            d_k = d[k:k + jb]
            e_k = e
            d[k:k + jb] = d_k - T.T @ d_k - T_e
            e = \
                -V[:, k:k + jb] @ T.T @ d_k + \
                e_k - \
                V[:, k:k + jb] @ T_e

            R_k = R[k: k + jb, k + jb:]
            U_k = U[:, k + jb:]

            R[k:k + jb, k + jb:] = R_k - T.T @ R_k - T_u
            U[:, k + jb:n] = -V[:, k:k + jb] @ T.T @ R_k + U_k - V[:p, k:k + jb] @ T_u

    R_tilde = np.zeros((m + p, n))
    R_tilde[:m] = R
    d_tilde = np.zeros((m + p, 1))
    d_tilde[:m] = d
    d_tilde[m:] = e
    resid = np.linalg.norm(d_tilde[n:], ord=2)

    ###################################
    # Algorithm 2.9 - compute Q_tilde.
    # Note: This part is not blocked
    # (upper part - for R - is).
    ###################################

    Q_tilde = np.zeros((m + p, m + p))
    Q_tilde[:m, :m] = Q
    Q_tilde[m:, m:] = np.eye(p, p)

    if k != m:
        insert_pos = k if k == 0 else k - 1
        Q_tilde[insert_pos:insert_pos + p] = Q_tilde[m:m + p, :]
        Q_tilde[k + p:] = Q_tilde[k:m, :]

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
    # Algorithm 2.23 - compute R_tilde.
    ###################################

    U = Q.T @ U
    d_tilde = Q.T @ b

    if m > n + 1:
        Q_u, R_u = qr_decomposition(U[n:m, :])
        d_tilde[n :m] = Q_u.T @ d_tilde[n:m]

    if k <= n:
        for j in np.arange(start=0, stop=p , step=1):
            upfirst = n - 1
            for i in np.arange(start=n + j, stop=j, step=-1):
                C[i, j], S[i, j] = givens(U[i - 1, j], U[i, j])
                cs_matrix = np.asarray([[C[i, j], C[i, j]], [-S[i, j], C[i, j]]])
                U[i - 1, j] = C[i, j] * U[i - 1, j] - S[i, j] * U[i, j]

                if j < p - 1:
                    U[i - 1:i + 1, j + 1:p] = cs_matrix.T @ U[i - 1:i + 1, j + 1:p]

                R[i - 1: i + 1, upfirst:n] = cs_matrix.T @ R[i - 1: i + 1, upfirst:n]
                upfirst -= 1
                d_tilde[i - 1:i + 1] = cs_matrix.T @ d_tilde[i - 1:i + 1]

    R_tilde = np.zeros((m, p + n))
    if k == 0:
        R_tilde[:, :p] = U
        R_tilde[:, p:] = R
    elif k == n:
        R_tilde[:, :n] = R
        R_tilde[:, n:] = U
    else:
        R_tilde[:, :k] = R[:, :k]
        R_tilde[:, k:k + p] = U
        R_tilde[:, k + p:] = R[:, k:]
    R_tilde = np.triu(R_tilde)

    resid = np.linalg.norm(d_tilde[n:], ord=2)

    ###################################
    # Algorithm 2.22 - compute Q_tilde.
    ###################################

    if m > n + 1:
        Q[:, :m - n] = Q[:, :m - n] @ Q_u

    if k <= n - 1:
        for j in np.arange(start=0, stop=p, step=1):
            for i in np.arange(start=n + j, stop=j, step=-1):
                cs_matrix = np.asarray([[C[i, j], C[i, j]], [-S[i, j], C[i, j]]])
                Q[:, i - 1:i + 1] = Q[:, i - 1:i + 1] @ cs_matrix

    return Q, R_tilde, resid
