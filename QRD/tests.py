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
        Q_corr, R_corr = np.linalg.qr(A)
        duration_lib = time.time() - start

        # With own implementation.
        start = time.time()
        Q, R = alg.qr_decomposition(A)
        duration_own = time.time() - start

        #################################################################
        # Gather evaluation data.
        #################################################################

        results["time_own"].append(duration_own)
        results["time_lib"].append(duration_lib)
        results["m"].append(m)
        results["n"].append(n)
        results["mn"].append(m * n)
        results["res_norm_QR"].append(alg.compute_residual(alg.matmul(Q, R), A))
        results["res_norm_Axb"].append(alg.compute_residual(scipy.linalg.solve(R, np.dot(Q.T, b)), x))

        pbar.update(m * n)
    pbar.close()

    return pd.DataFrame(results)


def test_row_deletion(size: tuple, fraction_to_remove: float) -> pd.DataFrame:
    """
    Tests deletion of rows.
    Blocked removal currently not supported.
    :param size:
    :return:
    """
    assert 0 < fraction_to_remove < 1, "fraction_to_remove has to be > 0 and < 1."

    results = {
        "m": [],
        "n": [],
        "mn": [],
        "time_own": [],
        "time_lib": [],
        "res_norm_QR": [],
        "res_norm_Axb": []
    }

    m, n = size
    A = np.random.rand(m, n)

    # todo: compare single row updates with recalculation from scratch
    # using
    #   - (1) numpy.linalg,
    #   - (2) scipy.linalg.qr_delete(),
    #   - (3) own deletion alg.,
    #   - (4) own blocked deletion algorithm.
    # -> applicable to all row/col. add/del. test functions.
    # note that 1, 2, 4 are executed only once. single row qr_deltetion has to be wrapped.

    pbar = tqdm(total=int(m * (1 - fraction_to_remove)))
    for size in sizes:
        m, n = size

        ################################################################
        # Generate data and compute correct results for solving Ax = b
        # with x being a vector of ones..
        #################################################################

        x = np.ones((n, 1))
        b = np.dot(A, x)

        #################################################################
        # Decompose A in Q and R.
        #################################################################

        # With numpy.
        start = time.time()
        Q_corr, R_corr = np.linalg.qr(A)
        duration_lib = time.time() - start

        # With own implementation.
        start = time.time()
        Q, R = alg.qr_decomposition(A)
        duration_own = time.time() - start

        #################################################################
        # Gather evaluation data.
        #################################################################

        results["time_own"].append(duration_own)
        results["time_lib"].append(duration_lib)
        results["m"].append(m)
        results["n"].append(n)
        results["mn"].append(m * n)
        results["res_norm_QR"].append(alg.compute_residual(alg.matmul(Q, R), A))
        results["res_norm_Axb"].append(alg.compute_residual(scipy.linalg.solve(R, np.dot(Q.T, b)), x))

        pbar.update(m * n)
    pbar.close()

    return pd.DataFrame(results)
