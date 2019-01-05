import numpy as np
from algorithms import l1 as alg1, l2 as alg2, l3 as alg3
import time
from tqdm import tqdm
import scipy
import scipy.linalg._decomp_update as scipy_qr_update
import pandas as pd
import utils
from typing import Tuple


def test_del_rows(size: tuple, row_sizes: list) -> pd.DataFrame:
    """
    Tests deletion of rows.
    :param size:
    :param row_sizes:
    :return:
    """

    results = {}
    m, n = size
    A = np.random.rand(m, n)
    x = np.ones((n, 1))
    b = np.dot(A, x)
    Q, R = scipy.linalg.qr(A)

    pbar = tqdm(total=n * sum(row_sizes))
    for m_tilde in row_sizes:
        ################################################################
        # Generate data and compute correct results for solving Ax = b
        # with x being a vector of ones..
        #################################################################

        assert m_tilde < m, "Make sure that m_tilde < m."
        A_tilde = A[m - m_tilde:m, :]
        p = m - m_tilde
        x_tilde = np.ones((n, 1))
        b_tilde_corr = np.dot(A_tilde, x_tilde)

        #################################################################
        # Remove rows.
        #################################################################

        # With recalculating from scratch with numpy().
        Q_tilde_corr, R_tilde_corr = compute_reference_QR(A_tilde, results)

        # With own implementation from scratch.
        compute_own_QR(A_tilde, Q_tilde_corr, R_tilde_corr, x_tilde, b_tilde_corr, results)

        # With scipy's QR update.
        start = time.time()
        Q_tilde, R_tilde = scipy_qr_update.qr_delete(Q, R, k=0, p=p, which="row")
        utils.update_measurements(
            results, "scipy_update", time.time() - start, A, A_tilde, x_tilde, b_tilde_corr,
            Q_tilde[:, :n], np.triu(R_tilde[:n]), op="del_rows", p=p, k=0
        )

        # With own L1 implementation.
        Q_tilde, R_tilde, b_tilde = np.copy(Q), np.copy(R), np.copy(b)
        start = time.time()
        for i in range(0, p):
            Q_tilde, R_tilde, b_tilde, residual = alg1.qr_delete_row(Q_tilde, R_tilde, b_tilde, k=0)
        utils.update_measurements(
            results, "l1", time.time() - start, A, A_tilde, x_tilde, b_tilde_corr, Q_tilde[:, :n], np.triu(R_tilde[:n]),
            op="del_rows", p=p, k=0
        )

        # With own L2 implementation.
        Q_input, R_input, b_tilde = np.copy(Q), np.copy(R), np.copy(b)
        start = time.time()
        Q_tilde, R_tilde, b_tilde, residual = alg2.qr_delete_rows(Q_input, R_input, b, p=p, k=0)
        utils.update_measurements(
            results, "l2", time.time() - start, A, A_tilde, x_tilde, b_tilde_corr, Q_tilde[:, :n], np.triu(R_tilde[:n]),
            op="del_rows", p=p, k=0
        )

        pbar.update(m_tilde * n)
    pbar.close()

    return pd.DataFrame.from_dict(results, orient="index")


def test_add_rows(size: tuple, row_sizes: list) -> pd.DataFrame:
    """
    Tests addition of rows.
    :param size:
    :param row_sizes:
    :return:
    """

    results = {}
    m, n = size
    A = np.random.rand(m, n)
    x = np.ones((n, 1))
    b = np.dot(A, x)
    Q, R = scipy.linalg.qr(A)

    pbar = tqdm(total=n * sum(row_sizes))
    for m_tilde in row_sizes:
        ################################################################
        # Generate data and compute correct results for solving Ax = b
        # with x being a vector of ones..
        #################################################################

        assert m_tilde > m, "Make sure that m_tilde < m."
        A_tilde = np.zeros((m_tilde, n))
        A_tilde[:m] = A
        U = np.random.rand(m_tilde - m, n)
        A_tilde[m:] = U
        x_tilde = np.ones((n, 1))
        b_tilde_corr = np.dot(A_tilde, x_tilde)

        #################################################################
        # Remove rows.
        #################################################################

        # With recalculating from scratch with numpy().
        Q_tilde_corr, R_tilde_corr = compute_reference_QR(A_tilde, results)

        # With own implementation from scratch.
        compute_own_QR(A_tilde, Q_tilde_corr, R_tilde_corr, x_tilde, b_tilde_corr, results)

        # With scipy's QR update.
        start = time.time()
        Q_tilde, R_tilde = scipy_qr_update.qr_insert(Q, R, k=m, u=U, which="row")
        # print(np.allclose(np.dot(Q_tilde_corr, R_tilde_corr), np.dot(Q_tilde_update, R_tilde_update)))
        utils.update_measurements(
            results, "scipy_update", time.time() - start, A, A_tilde, x_tilde, b_tilde_corr, Q_tilde[:, :n],
            np.triu(R_tilde[:n]), op="add_rows", p=m_tilde - m, k=m
        )

        # With own L1 implementation.
        Q_tilde, R_tilde, b_tilde = np.copy(Q), np.copy(R), np.copy(b)
        start = time.time()
        for i in range(m, m_tilde, 1):
            Q_tilde, R_tilde, b_tilde, residual = alg1.qr_add_row(
                Q_tilde, R_tilde, u=U[i - m], b=b_tilde, mu=b_tilde_corr[i:][0], k=m
            )
        utils.update_measurements(
            results, "l1", time.time() - start, A, A_tilde, x_tilde, b_tilde_corr, Q_tilde[:, :n], np.triu(R_tilde[:n]),
            op="add_rows", p=m_tilde - m, k=m
        )

        # With own L2 implementation.
        Q_input, R_input = np.copy(Q), np.copy(R)
        start = time.time()
        Q_tilde, R_tilde, b_tilde, residual = alg2.qr_add_rows(
            Q_input, R_input, U=U, b=b, e=b_tilde_corr[m:m_tilde], k=m
        )
        utils.update_measurements(
            results, "l2", time.time() - start, A, A_tilde, x_tilde, b_tilde_corr, Q_tilde[:, :n], np.triu(R_tilde[:n]),
            op="add_rows", p=m_tilde - m, k=m
        )

        # With own L3 implementation.
        Q_input, R_input = np.copy(Q), np.copy(R)
        start = time.time()
        Q_tilde, R_tilde, b_tilde, residual = alg3.qr_add_rows(
            Q_input, R_input, U=U, b=b, e=b_tilde_corr[m:m_tilde], k=m
        )
        utils.update_measurements(
            results, "l3", time.time() - start, A, A_tilde, x_tilde, b_tilde_corr, Q_tilde[:, :n], np.triu(R_tilde[:n]),
            op="add_rows", p=m_tilde - m, k=m
        )

        pbar.update(m_tilde * n)
    pbar.close()

    return pd.DataFrame.from_dict(results, orient="index")


def test_del_cols(size: tuple, col_sizes: list) -> pd.DataFrame:
    """
    Tests deletion of columns.
    :param size:
    :param col_sizes:
    :return:
    """

    results = {}
    m, n = size
    A = np.random.rand(m, n)
    x = np.ones((n, 1))
    b = np.dot(A, x)
    Q, R = scipy.linalg.qr(A)

    pbar = tqdm(total=m * sum(col_sizes))
    for n_tilde in col_sizes:
        ################################################################
        # Generate data and compute correct results for solving Ax = b
        # with x being a vector of ones..
        #################################################################

        assert n_tilde < n, "Make sure that n_tilde < n."
        A_tilde = A[:, n - n_tilde:]
        x_tilde = np.ones((n_tilde, 1))
        b_tilde_corr = np.dot(A_tilde, x_tilde)
        p = n - n_tilde

        #################################################################
        # Remove columns.
        #################################################################

        # With recalculating from scratch with numpy().
        Q_tilde_corr, R_tilde_corr = compute_reference_QR(A_tilde, results)

        # With own implementation from scratch.
        compute_own_QR(A_tilde, Q_tilde_corr, R_tilde_corr, x_tilde, b_tilde_corr, results)

        # With scipy's QR update.
        start = time.time()
        Q_tilde, R_tilde = scipy_qr_update.qr_delete(Q, R, k=0, p=p, which="col")
        utils.update_measurements(
            results, "scipy_update", time.time() - start, A, A_tilde, x_tilde, b_tilde_corr, Q_tilde.T[:n_tilde].T,
            np.triu(R_tilde[:n_tilde]), op="del_cols", p=p, k=0
        )

        # With own L1 implementation..
        Q_tilde, R_tilde, b_tilde = np.copy(Q), np.copy(R), np.copy(b)
        start = time.time()
        for i in range(0, n - n_tilde):
            Q_tilde, R_tilde, residual = alg1.qr_delete_col(Q_tilde, R_tilde, b_tilde, k=0)
        utils.update_measurements(
            results, "l1", time.time() - start, A, A_tilde, x_tilde, b_tilde_corr, Q_tilde.T[:n_tilde].T,
            np.triu(R_tilde[:n_tilde]), op="del_cols", p=p, k=0
        )

        # With own L2 implementation.
        Q_input, R_input, b_tilde = np.copy(Q), np.copy(R), np.copy(b)
        start = time.time()
        Q_tilde, R_tilde, residual = alg2.qr_delete_cols(Q_input, R_input, b, p=p, k=0)
        utils.update_measurements(
            results, "l2", time.time() - start, A, A_tilde, x_tilde, b_tilde_corr, Q_tilde.T[:n_tilde].T,
            np.triu(R_tilde[:n_tilde]), op="del_cols", p=p, k=0
        )

        pbar.update(m * n_tilde)
    pbar.close()

    return pd.DataFrame.from_dict(results, orient="index")


def test_add_cols(size: tuple, col_sizes: list) -> pd.DataFrame:
    """
    Tests addition of columns.
    :param size:
    :param col_sizes:
    :return:
    """

    results = {}

    m, n = size
    A = np.random.rand(m, n)
    x = np.ones((n, 1))
    b = np.dot(A, x)
    Q, R = scipy.linalg.qr(A)

    pbar = tqdm(total=m * sum(col_sizes))
    for n_tilde in col_sizes:
        ################################################################
        # Generate data and compute correct results for solving Ax = b
        # with x being a vector of ones..
        #################################################################

        assert n_tilde > n, "Make sure that n_tilde > n."
        A_tilde = np.zeros((m, n_tilde))
        A_tilde[:, :n] = A
        U = np.random.rand(m, n_tilde - n)
        A_tilde[:, n:] = U
        p = n_tilde - n
        x_tilde = np.ones((n_tilde, 1))
        b_tilde_corr = np.dot(A_tilde, x_tilde)

        #################################################################
        # Remove rows.
        #################################################################

        # With recalculating from scratch with numpy().
        Q_tilde_corr, R_tilde_corr = compute_reference_QR(A_tilde, results)

        # With own implementation from scratch.
        compute_own_QR(A_tilde, Q_tilde_corr, R_tilde_corr, x_tilde, b_tilde_corr, results)

        # With scipy's QR update.
        start = time.time()
        Q_tilde, R_tilde = scipy_qr_update.qr_insert(Q, R, k=n, u=U, which="col")
        utils.update_measurements(
            results, "scipy_update", time.time() - start, A, A_tilde, x_tilde, b_tilde_corr, Q_tilde.T[:n + p].T,
            np.triu(R_tilde[:n + p]), op="add_cols", p=p, k=n
        )

        # With own L1 implementation.
        Q_tilde, R_tilde = np.copy(Q), np.copy(R)
        start = time.time()
        for i in range(0, p):
            Q_tilde, R_tilde, residual = alg1.qr_add_col(Q_tilde, R_tilde, u=U[:, i], b=b, k=n)
        utils.update_measurements(
            results, "l1", time.time() - start, A, A_tilde, x_tilde, b_tilde_corr, Q_tilde.T[:n + p].T,
            np.triu(R_tilde[:n + p]), op="add_cols", p=p, k=n
        )

        # With own L2 implementation.
        Q_input, R_input = np.copy(Q), np.copy(R)
        start = time.time()
        Q_tilde, R_tilde, residual = alg2.qr_add_cols(Q_input, R_input, U=U, p=p, b=b, k=0)
        utils.update_measurements(
            results, "l2", time.time() - start, A, A_tilde, x_tilde, b_tilde_corr, Q_tilde.T[:n + p].T,
            np.triu(R_tilde[:n + p]), op="add_cols", p=p, k=n
        )

        # With own L3 implementation.
        Q_input, R_input = np.copy(Q), np.copy(R)
        start = time.time()
        Q_tilde, R_tilde, residual = alg3.qr_add_cols(Q_input, R_input, U=U, p=p, b=b, k=0)
        utils.update_measurements(
            results, "l3", time.time() - start, A, A_tilde, x_tilde, b_tilde_corr, Q_tilde.T[:n + p].T,
            np.triu(R_tilde[:n + p]), op="add_cols", p=p, k=n
        )

        pbar.update(m * n_tilde)

    pbar.close()

    return pd.DataFrame.from_dict(results, orient="index")


def compute_reference_QR(A: np.ndarray, results: dict):
    """
    Computes reference QR decomposition. Includes measurement updates.
    :param A:
    :param results:
    :return:
    """
    start = time.time()
    Q, R = scipy.linalg.qr(A)
    utils.update_measurements(
        results, "scipy_scratch", time.time() - start, A, A, compute_accuracy=False, op="scratch"
    )

    return Q, R


def compute_own_QR(
        A: np.ndarray, Q_prime: np.ndarray, R_prime: np.ndarray, x_prime: np.ndarray, b_prime: np.ndarray, results: dict
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes QR decomposition with own method.
    :param A:
    :param A_prime:
    :param results:
    :return:
    """

    m, n = A.shape

    start = time.time()
    Q_tilde, R_tilde = alg1.qr_decomposition(A)
    utils.update_measurements(
        results, "l2_scratch", time.time() - start, A, A,
        x_prime, b_prime, Q_tilde[:, :n], np.triu(R_tilde[:n]), op="scratch"
    )

    return Q_tilde, R_tilde
