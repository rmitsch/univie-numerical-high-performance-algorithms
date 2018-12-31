import numpy as np
import algorithms as alg
import time
from tqdm import tqdm
import scipy
import scipy.linalg._decomp_update as scipy_qr_update
import pandas as pd


def test_generation_from_scratch(sizes: list) -> pd.DataFrame:
    """
    Tests generation of Q, R from scratch with a variety of sizes.
    :param sizes:
    :return:
    """
    results = {
        "m": [],
        "n": [],
        "mn": [],
        "time_own": [],
        "time_lib": [],
        "res_norm_QR": [],
        "res_norm_Axb": []
    }

    pbar = tqdm(total=np.sum(m * n for m, n in sizes))
    for size in sizes:
        m, n = size

        ################################################################
        # Generate data and compute correct results for solving Ax = b
        # with x being a vector of ones..
        #################################################################

        A = np.random.rand(m, n)
        x = np.ones((n, 1))
        b = np.dot(A, x)

        #################################################################
        # Decompose A in Q and R.
        #################################################################

        # With numpy.
        start = time.time()
        scipy.linalg.qr(A)
        duration_lib = time.time() - start

        # With own implementation.
        start = time.time()
        Q, R = alg.qr_decomposition(A)
        duration_own = time.time() - start

        #################################################################
        # Gather evaluation data.
        #################################################################

        Q_eval = Q.T[:n].T
        R_eval = np.triu(R[:n])
        results["time_own"].append(duration_own)
        results["time_lib"].append(duration_lib)
        results["m"].append(m)
        results["n"].append(n)
        results["mn"].append(m * n)
        results["res_norm_QR"].append(alg.compute_residual(alg.matmul(Q_eval, R_eval), A))
        results["res_norm_Axb"].append(alg.compute_residual(scipy.linalg.solve(R_eval, np.dot(Q_eval.T, b)), x))

        pbar.update(m * n)
    pbar.close()

    return pd.DataFrame(results)


def test_row_deletion(size: tuple, modified_sizes: list) -> pd.DataFrame:
    """
    Tests deletion of rows.
    Blocked removal currently not supported.
    :param size:
    :param modified_sizes:
    :return:
    """

    results = {
        "m": [],
        "n": [],
        "mn": [],
        "time_own": [],
        "time_own_blocked": [],
        "time_scipy_update": [],
        "time_numpy_scratch": [],
        "res_norm_QR": [],
        "res_norm_Axb": []
    }

    m, n = size
    A = np.random.rand(m, n)
    x = np.ones((n, 1))
    b = np.dot(A, x)
    Q, R = scipy.linalg.qr(A)
    print(Q.shape, R.shape)

    # todo: compare single row updates with recalculation from scratch and scipy's row updates
    # using
    #   - (1) numpy.linalg,
    #   - (2) scipy.linalg.qr_delete(),
    #   - (3) own deletion alg.,
    #   - (4) own blocked deletion algorithm.
    # -> applicable to all row/col. add/del. test functions.
    # note that 1, 2, 4 are executed only once. single row qr_deltetion has to be wrapped.
    # with/without numba are two separate executions of test sequence.

    pbar = tqdm(total=(m * len(modified_sizes) - sum([m for m, n in modified_sizes])))
    pbar.close()
    for size_tilde in modified_sizes:
        m_tilde, n_tilde = size_tilde
        A_tilde = A[m - m_tilde:m, :]

        ################################################################
        # Generate data and compute correct results for solving Ax = b
        # with x being a vector of ones..
        #################################################################

        x_tilde = np.ones((n_tilde, 1))
        b_tilde_corr = np.dot(A_tilde, x_tilde)

        #################################################################
        # Remove rows.
        #################################################################

        # With recalculating from scratch with numpy().
        start = time.time()
        Q_tilde_corr, R_tilde_corr = scipy.linalg.qr(A_tilde)
        duration_numpy_scratch = time.time() - start

        # With scipy's qr_remove().
        start = time.time()
        Q_tilde_corr, R_tilde_corr = scipy_qr_update.qr_delete(Q, R, k=0, p=m - m_tilde, which="row")
        # print(np.allclose(np.dot(Q_tilde_corr, R_tilde_corr), np.dot(Q_tilde_update, R_tilde_update)))
        duration_scipy_update = time.time() - start

        # With own implementation.
        start = time.time()
        Q_input, R_input = np.copy(Q), np.copy(R)
        b_tilde = np.copy(b)
        # for i in range(0, m - m_tilde):
        Q_tilde, R_tilde, b_tilde, residual = alg.qr_delete_row(Q_input, R_input, b, k=0)
        duration_own = time.time() - start

        print(Q_tilde.shape, Q_tilde_corr.shape)
        print(R_tilde.shape, R_tilde_corr.shape)
        print(np.allclose(np.dot(Q_tilde_corr, R_tilde_corr), np.dot(Q_tilde, R_tilde)))
        print(np.allclose(Q_tilde_corr, Q_tilde), np.allclose(R_tilde_corr, R_tilde))
        print(np.allclose(b_tilde_corr, b_tilde))

        #################################################################
        # Gather evaluation data.
        #################################################################

        Q_tilde_eval = Q_tilde.T[:n_tilde].T
        R_tilde_eval = np.triu(R_tilde[:n_tilde])

        print(alg.compute_residual(
            scipy.linalg.solve(R_tilde_eval, np.dot(Q_tilde_eval.T, b_tilde)), x_tilde
        ))
        exit()

        results["time_own"].append(duration_own)
        results["time_blocked"].append(0)
        results["time_numpy_scratch"].append(duration_numpy_scratch)
        results["time_scipy_update"].append(duration_scipy_update)
        results["m"].append(m)
        results["n"].append(n)
        results["mn"].append(m * n)
        results["res_norm_QR"].append(alg.compute_residual(
            alg.matmul(Q_tilde_eval, R_tilde_eval), A_tilde
        ))
        results["res_norm_Axb"].append(alg.compute_residual(
            scipy.linalg.solve(R_tilde_eval, np.dot(Q_tilde_eval.T, b_tilde)), x_tilde
        ))

        pbar.update(m * n)
    pbar.close()

    return pd.DataFrame(results)
