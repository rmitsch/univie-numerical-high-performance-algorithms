import numpy as np
import algorithms as alg
import time
from tqdm import tqdm
import scipy
import scipy.linalg._decomp_update as scipy_qr_update
import pandas as pd
import utils


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


def test_del_row(size: tuple, modified_sizes: list) -> pd.DataFrame:
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
        m_tilde, _ = size_tilde
        assert m_tilde < m, "Make sure that m_tilde < m."
        A_tilde = A[m - m_tilde:m, :]

        ################################################################
        # Generate data and compute correct results for solving Ax = b
        # with x being a vector of ones..
        #################################################################

        x_tilde = np.ones((n, 1))
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

        Q_tilde_eval = Q_tilde.T[:n].T
        R_tilde_eval = np.triu(R_tilde[:n])

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


def test_add_row(size: tuple, modified_sizes: list) -> pd.DataFrame:
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

    pbar = tqdm(total=(m * len(modified_sizes) - sum([m for m, n in modified_sizes])))
    pbar.close()
    for size_tilde in modified_sizes:
        m_tilde, _ = size_tilde
        assert m_tilde > m, "Make sure that m_tilde < m."
        A_tilde = np.zeros((m_tilde, n))
        A_tilde[:m] = A
        u = np.random.rand(m_tilde - m, n)
        A_tilde[m:] = u

        ################################################################
        # Generate data and compute correct results for solving Ax = b
        # with x being a vector of ones..
        #################################################################

        x_tilde = np.ones((n, 1))
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
        Q_tilde_corr, R_tilde_corr = scipy_qr_update.qr_insert(Q, R, k=m, u=u, which="row")
        # print(np.allclose(np.dot(Q_tilde_corr, R_tilde_corr), np.dot(Q_tilde_update, R_tilde_update)))
        duration_scipy_update = time.time() - start

        # With own implementation.
        start = time.time()
        Q_input, R_input = np.copy(Q), np.copy(R)

        # for i in range(0, m - m_tilde):
        Q_tilde, R_tilde, b_tilde, residual = alg.qr_add_row(Q_input, R_input, u=u[0], b=b, mu=b_tilde_corr[m:][0], k=m)
        duration_own = time.time() - start

        print(Q_tilde.shape, Q_tilde_corr.shape)
        print(R_tilde.shape, R_tilde_corr.shape)
        print(np.allclose(np.dot(Q_tilde_corr, R_tilde_corr), np.dot(Q_tilde, R_tilde)))
        print(np.allclose(Q_tilde_corr, Q_tilde), np.allclose(R_tilde_corr, R_tilde))
        print(np.allclose(b_tilde_corr, b_tilde))

        #################################################################
        # Gather evaluation data.
        #################################################################

        Q_tilde_eval = Q_tilde.T[:n].T
        R_tilde_eval = np.triu(R_tilde[:n])

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


def test_del_col(size: tuple, modified_sizes: list) -> pd.DataFrame:
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

    pbar = tqdm(total=(n * len(modified_sizes) - sum([n for m, n in modified_sizes])))
    pbar.close()
    for size_tilde in modified_sizes:
        ################################################################
        # Generate data and compute correct results for solving Ax = b
        # with x being a vector of ones..
        #################################################################

        _, n_tilde = size_tilde
        assert n_tilde < n, "Make sure that n_tilde < n."
        A_tilde = A[:, n - n_tilde:]
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
        Q_tilde_corr, R_tilde_corr = scipy_qr_update.qr_delete(Q, R, k=0, p=n - n_tilde, which="col")
        # print(np.allclose(np.dot(Q_tilde_corr, R_tilde_corr), np.dot(Q_tilde_update, R_tilde_update)))
        duration_scipy_update = time.time() - start

        # With own implementation.
        start = time.time()
        Q_input, R_input = np.copy(Q), np.copy(R)
        # for i in range(0, m - m_tilde):
        Q_tilde, R_tilde, residual = alg.qr_delete_col(Q_input, R_input, b, k=0)
        duration_own = time.time() - start

        print(Q_tilde.shape, Q_tilde_corr.shape)
        print(R_tilde.shape, R_tilde_corr.shape)
        print(np.allclose(np.dot(Q_tilde_corr, R_tilde_corr), np.dot(Q_tilde, R_tilde)))
        print(np.allclose(Q_tilde_corr, Q_tilde), np.allclose(R_tilde_corr, R_tilde))

        #################################################################
        # Gather evaluation data.
        #################################################################

        Q_tilde_eval = Q_tilde.T[:n - 1].T
        R_tilde_eval = np.triu(R_tilde[:n - 1])

        print(alg.compute_residual(
            scipy.linalg.solve(R_tilde_eval, np.dot(Q_tilde_eval.T, b_tilde_corr)), x_tilde
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
            scipy.linalg.solve(R_tilde_eval, np.dot(Q_tilde_eval.T, b_tilde_corr)), x_tilde
        ))

        pbar.update(m * n)
    pbar.close()

    return pd.DataFrame(results)


def test_add_col(size: tuple, modified_sizes: list) -> pd.DataFrame:
    """
    Tests deletion of rows.
    Blocked removal currently not supported.
    :param size:
    :param modified_sizes:
    :return:
    """

    results = utils.init_result_dict()

    m, n = size
    A = np.random.rand(m, n)
    x = np.ones((n, 1))
    b = np.dot(A, x)
    Q, R = scipy.linalg.qr(A)

    pbar = tqdm(total=(m * len(modified_sizes) - sum([m for m, n in modified_sizes])))
    pbar.close()
    for size_tilde in modified_sizes:
        _, n_tilde = size_tilde
        assert n_tilde > n, "Make sure that n_tilde > n."
        A_tilde = np.zeros((m, n_tilde))
        A_tilde[:, :n] = A
        u = np.random.rand(m, n_tilde - n)
        A_tilde[:, n:] = u

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
        Q_tilde_corr, R_tilde_corr = scipy_qr_update.qr_insert(Q, R, k=0, u=u, which="col")
        # print(np.allclose(np.dot(Q_tilde_corr, R_tilde_corr), np.dot(Q_tilde_update, R_tilde_update)))
        duration_scipy_update = time.time() - start

        # With own implementation.
        start = time.time()
        Q_input, R_input = np.copy(Q), np.copy(R)

        # for i in range(0, m - m_tilde):
        Q_tilde, R_tilde, residual = alg.qr_add_col(Q_input, R_input, u=u[:, 0], b=b, k=n)
        duration_own = time.time() - start

        print(Q_tilde.shape, Q_tilde_corr.shape)
        print(R_tilde.shape, R_tilde_corr.shape)
        print(np.allclose(np.dot(Q_tilde_corr, R_tilde_corr), np.dot(Q_tilde, R_tilde)))
        print(np.allclose(Q_tilde_corr, Q_tilde), np.allclose(R_tilde_corr, R_tilde))

        #################################################################
        # Gather evaluation data.
        #################################################################

        Q_tilde_eval = Q_tilde.T[:n + 1].T
        R_tilde_eval = np.triu(R_tilde[:n + 1])

        print(alg.compute_residual(
            scipy.linalg.solve(R_tilde_eval, np.dot(Q_tilde_eval.T, b_tilde_corr)), x_tilde
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
            scipy.linalg.solve(R_tilde_eval, np.dot(Q_tilde_eval.T, b_tilde_corr)), x_tilde
        ))

        pbar.update(m * n)
    pbar.close()

    return pd.DataFrame(results)
