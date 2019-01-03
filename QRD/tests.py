import numpy as np
from algorithms import l1 as alg1, l2 as alg2, l3 as alg3
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

        # With own implementation..
        start = time.time()
        Q, R = alg1.qr_decomposition(A)
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
        results["res_norm_QR"].append(alg1.compute_residual(alg1.matmul(Q_eval, R_eval), A))
        results["res_norm_Axb"].append(alg1.compute_residual(scipy.linalg.solve(R_eval, np.dot(Q_eval.T, b)), x))

        pbar.update(m * n)
    pbar.close()

    return pd.DataFrame(results)


def test_del_rows(size: tuple, modified_sizes: list) -> pd.DataFrame:
    """
    Tests deletion of rows.
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
        m_tilde, _ = size_tilde
        assert m_tilde < m, "Make sure that m_tilde < m."
        A_tilde = A[m - m_tilde:m, :]
        p = m - m_tilde

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
        results["time_numpy_scratch"].append(time.time() - start)

        # With scipy's qr_remove().
        start = time.time()
        Q_tilde_update, R_tilde_update = scipy_qr_update.qr_delete(Q, R, k=0, p=p, which="row")
        results["time_scipy_update"].append(time.time() - start)

        # With own L1 implementation..
        start = time.time()
        Q_tilde, R_tilde, b_tilde = np.copy(Q), np.copy(R), np.copy(b)
        for i in range(0, p):
            Q_tilde, R_tilde, b_tilde, residual = alg1.qr_delete_row(Q_tilde, R_tilde, b_tilde, k=0)
        results["time_own"].append(time.time() - start)

        # With own L2 implementation.
        start = time.time()
        Q_input, R_input = np.copy(Q), np.copy(R)
        b_tilde = np.copy(b)
        Q_tilde, R_tilde, b_tilde, residual = alg2.qr_delete_rows(Q_input, R_input, b, p=p, k=0)
        results["time_own_blocked"].append(time.time() - start)

        print(b_tilde.shape, b_tilde_corr.shape)
        print(Q_tilde.shape, Q_tilde_corr.shape)
        print(R_tilde.shape, R_tilde_corr.shape)
        print(np.allclose(np.dot(Q_tilde_corr, R_tilde_corr), np.dot(Q_tilde, R_tilde)))
        print(np.allclose(Q_tilde_corr, Q_tilde), np.allclose(R_tilde_corr, R_tilde))
        print(np.allclose(b_tilde_corr, b_tilde))

        #################################################################
        # Gather evaluation data.
        #################################################################

        Q_tilde_eval = Q_tilde[:, :n]
        R_tilde_eval = np.triu(R_tilde[:n])

        print(R_tilde_eval.shape, Q_tilde_eval.shape, np.dot(Q_tilde_eval.T, b_tilde).shape)

        # print(scipy.linalg.solve(R_tilde_eval, np.dot(Q_tilde_eval.T, b_tilde)))
        try:
            print(alg1.compute_residual(
                scipy.linalg.solve(R_tilde_eval, np.dot(Q_tilde_eval.T, b_tilde)), x_tilde
            ))
        except np.linalg.LinAlgError as e:
            print("Singular matrix.")
        exit()

        results["m"].append(m_tilde)
        results["n"].append(n)
        results["mn"].append(m_tilde * n)
        results["res_norm_QR"].append(alg1.compute_residual(
            alg1.matmul(Q_tilde_eval, R_tilde_eval), A_tilde
        ))
        results["res_norm_Axb"].append(alg1.compute_residual(
            scipy.linalg.solve(R_tilde_eval, np.dot(Q_tilde_eval.T, b_tilde)), x_tilde
        ))

        pbar.update(m * n)
    pbar.close()

    return pd.DataFrame(results)


def test_add_rows(size: tuple, modified_sizes: list) -> pd.DataFrame:
    """
    Tests addition of rows.
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
        m_tilde, _ = size_tilde
        assert m_tilde > m, "Make sure that m_tilde < m."
        A_tilde = np.zeros((m_tilde, n))
        A_tilde[:m] = A
        U = np.random.rand(m_tilde - m, n)
        A_tilde[m:] = U

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
        results["time_numpy_scratch"].append(time.time() - start)

        # With scipy's qr_remove().
        start = time.time()
        Q_tilde_corr, R_tilde_corr = scipy_qr_update.qr_insert(Q, R, k=m, u=U, which="row")
        # print(np.allclose(np.dot(Q_tilde_corr, R_tilde_corr), np.dot(Q_tilde_update, R_tilde_update)))
        results["time_scipy_update"].append(time.time() - start)

        # With own L1 implementation..
        start = time.time()
        Q_tilde, R_tilde = np.copy(Q), np.copy(R)
        for i in range(m_tilde - m, 0):
            Q_tilde, R_tilde, b_tilde, residual = alg1.qr_add_row(
                Q_tilde, R_tilde, u=U[i], b=b, mu=b_tilde_corr[m:][i], k=0
            )
        results["time_own"].append(time.time() - start)

        # With own L2 implementation.
        start = time.time()
        Q_input, R_input = np.copy(Q), np.copy(R)
        Q_tilde, R_tilde, b_tilde, residual = alg2.qr_add_rows(
            Q_input, R_input, p=m_tilde - m, U=U, b=b, e=b_tilde_corr[m:m_tilde], k=m
        )
        results["time_own_blocked"].append(time.time() - start)

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

        print(alg1.compute_residual(
            scipy.linalg.solve(R_tilde_eval, np.dot(Q_tilde_eval.T, b_tilde)), x_tilde
        ))
        exit()

        results["m"].append(m_tilde)
        results["n"].append(n)
        results["mn"].append(m_tilde * n)
        results["res_norm_QR"].append(alg1.compute_residual(
            alg1.matmul(Q_tilde_eval, R_tilde_eval), A_tilde
        ))
        results["res_norm_Axb"].append(alg1.compute_residual(
            scipy.linalg.solve(R_tilde_eval, np.dot(Q_tilde_eval.T, b_tilde)), x_tilde
        ))

        pbar.update(m * n)
    pbar.close()

    return pd.DataFrame(results)


def test_del_cols(size: tuple, modified_sizes: list) -> pd.DataFrame:
    """
    Tests deletion of columns.
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
        p = n - n_tilde

        #################################################################
        # Remove rows.
        #################################################################

        # With recalculating from scratch with numpy().
        start = time.time()
        Q_tilde_corr, R_tilde_corr = scipy.linalg.qr(A_tilde)
        results["time_numpy_scratch"].append(time.time() - start)

        # With scipy's qr_remove().
        start = time.time()
        scipy_qr_update.qr_delete(Q, R, k=0, p=p, which="col")
        results["time_scipy_update"].append(time.time() - start)

        # With own L1 implementation..
        start = time.time()
        Q_tilde, R_tilde, b_tilde = np.copy(Q), np.copy(R), np.copy(b)
        for i in range(0, n - n_tilde):
            Q_tilde, R_tilde, residual = alg1.qr_delete_col(Q_tilde, R_tilde, b_tilde, k=0)
        results["time_own"].append(time.time() - start)

        # With own L2 implementation.
        start = time.time()
        print(n, n_tilde)
        Q_tilde, R_tilde, residual = alg2.qr_delete_cols(
            np.copy(Q), np.copy(R), b, p=p, k=0
        )
        results["time_own_blocked"].append(time.time() - start)

        print(Q_tilde.shape, Q_tilde_corr.shape)
        print(R_tilde.shape, R_tilde_corr.shape)
        print(np.allclose(np.dot(Q_tilde_corr, R_tilde_corr), np.dot(Q_tilde, R_tilde)))
        print(np.allclose(Q_tilde_corr, Q_tilde), np.allclose(R_tilde_corr, R_tilde))

        #################################################################
        # Gather evaluation data.
        #################################################################

        Q_tilde_eval = Q_tilde.T[:n_tilde].T
        R_tilde_eval = np.triu(R_tilde[:n_tilde])

        print(alg1.compute_residual(
            scipy.linalg.solve(R_tilde_eval, np.dot(Q_tilde_eval.T, b_tilde_corr)), x_tilde
        ))
        exit()

        results["m"].append(m)
        results["n"].append(n)
        results["mn"].append(m * n)
        results["res_norm_QR"].append(alg1.compute_residual(
            alg1.matmul(Q_tilde_eval, R_tilde_eval), A_tilde
        ))
        results["res_norm_Axb"].append(alg1.compute_residual(
            scipy.linalg.solve(R_tilde_eval, np.dot(Q_tilde_eval.T, b_tilde_corr)), x_tilde
        ))

        pbar.update(m * n)
    pbar.close()

    return pd.DataFrame(results)


def test_add_cols(size: tuple, modified_sizes: list) -> pd.DataFrame:
    """
    Tests addition of columns.
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
        U = np.random.rand(m, n_tilde - n)
        A_tilde[:, n:] = U

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
        Q_tilde_corr, R_tilde_corr = scipy_qr_update.qr_insert(Q, R, k=0, u=U, which="col")
        # print(np.allclose(np.dot(Q_tilde_corr, R_tilde_corr), np.dot(Q_tilde_update, R_tilde_update)))
        duration_scipy_update = time.time() - start

        # With own L1 implementation..
        start = time.time()
        Q_input, R_input = np.copy(Q), np.copy(R)
        # for i in range(0, m - m_tilde):
        Q_tilde, R_tilde, residual = alg1.qr_add_col(Q_input, R_input, u=U[:, 0], b=b, k=n)
        duration_own = time.time() - start

        # With own L2 implementation..
        start = time.time()
        Q_input, R_input = np.copy(Q), np.copy(R)
        
        Q_tilde, R_tilde, residual = alg2.qr_add_cols(Q_input, R_input, U=U, p=n_tilde - n, b=b, k=n)
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

        print(alg1.compute_residual(
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
        results["res_norm_QR"].append(alg1.compute_residual(
            alg1.matmul(Q_tilde_eval, R_tilde_eval), A_tilde
        ))
        results["res_norm_Axb"].append(alg1.compute_residual(
            scipy.linalg.solve(R_tilde_eval, np.dot(Q_tilde_eval.T, b_tilde_corr)), x_tilde
        ))

        pbar.update(m * n)
    pbar.close()

    return pd.DataFrame(results)
