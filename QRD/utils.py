import logging
import numpy as np
import scipy
import pandas as pd
from algorithms import l1 as alg1


METHODS = ["l2_scratch", "l1", "l2", "l3", "scipy_scratch", "scipy_update"]
# todo get correct number of peak flops!
MACHINE_FLOPS = 3.29 * np.power(10, 12)


def compute_flops(m: int, n: int, p: int, k: int, method: str, op: str):
    """
    Computes number of flops depending on used method and operation.
    :param m: 
    :param n:
    :param p:
    :param k:
    :param method: 
    :param op: 
    :return: 
    """
    k += 1

    if op == "del_rows":
        if method in ("l2_scratch", "scipy_scratch"):
            return 2 * n * n * (m - p - n / 3)
        elif method == "l1":
            return 3 * n * n
        elif method in ("l2", "scipy_update"):
            return 3 * n * n * p + p * p * (m / 3 - p)

    elif op == "add_rows":
        if method in ("l2_scratch", "scipy_scratch"):
            return 2 * n * n * (m + p - n / 3)
        elif method == "l1":
            return 3 * n * n
        elif method == "l2":
            return 2 * n * n * p
        elif method in ("l3", "scipy_update"):
            return 2 * n * n * p

    elif op == "del_cols":
        if method in ("l2_scratch", "scipy_scratch"):
            return 2 * (n - p) * (n - p) * (m - (n - p) / 3)
        elif method == "l1":
            return n * n / 2 - n * k + k * k / 2
        elif method in ("l2", "scipy_update"):
            return 4 * (n * p * (n / 2 - p - k) + p * p * (p / 2 + k) + p * k * k)

    elif op == "add_cols":
        if method in ("l2_scratch", "scipy_scratch"):
            return 2 * (n + p) * (n + p) * (m - (n + p) / 3)
        elif method == "l1":
            return n * n / 2 - n * k + k * k / 2
        elif method == "l2":
            return 6 * (m * p * (n + p - m / 2) - p * p * (n / 2 - k / 2 - p / 3) + k * p * (k / 2 - n))
        elif method in ("l3", "scipy_update"):
            return 6 * (m * p * (n + p - m / 2) - p * p * (n / 2 - k / 2 - p / 3) + k * p * (k / 2 - n))

    elif op == "scratch":
        return 2 * n * n * (m - n / 3)

    else:
        raise Exception("Method not supported")


def create_logger(name: str):
    """
    Initializes named logger.
    :param name:
    :return:
    """
    logging.basicConfig(
        level=logging.DEBUG, format="%(asctime)s : %(levelname)s : %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    return logging.getLogger(name)


def update_measurements(
        results: dict,
        method: str,
        time: float,
        A: np.ndarray,
        A_tilde: np.ndarray,
        x_tilde: np.ndarray = None,
        b_tilde: np.ndarray = None,
        Q_tilde_own: np.ndarray = None,
        R_tilde_own: np.ndarray = None,
        compute_accuracy: bool = True,
        op: str = None,
        p: int = 0,
        k: int = 0
):
    """
    Adds set of measurements to result dictionary.
    :param results:
    :param method:
    :param time:
    :param A_tilde:
    :param x_tilde:
    :param b_tilde:
    :param Q_tilde_own:
    :param R_tilde_own:
    :param compute_accuracy:
    :param op:
    :param p:
    :param k:
    :return:
    """
    m_tilde, n_tilde = A_tilde.shape
    m, n = A.shape
    key = (m_tilde, n_tilde)

    if key not in results:
        results[key] = {
            "m": m_tilde,
            "n": n_tilde,
            "mn": m_tilde * n_tilde
        }

        for meth in METHODS:
            results[key][meth + "_time"] = 0
            results[key][meth + "_rn_QR"] = 0
            results[key][meth + "_rn_Axb"] = 0
            results[key][meth + "_eff"] = 0

    results[key][method + "_time"] = time
    results[key][method + "_eff"] = compute_flops(
        m, n, p, k, method, op
    ) / time / MACHINE_FLOPS

    if compute_accuracy:
        try:
            results[key][method + "_rn_QR"] = alg1.compute_residual(
                alg1.matmul(Q_tilde_own, R_tilde_own), A_tilde
            )
        except np.linalg.LinAlgError:
            results[key][method + "_rn_QR"] = -1

        try:
            results[key][method + "_rn_Axb"] = alg1.compute_residual(
                scipy.linalg.solve(R_tilde_own, np.dot(Q_tilde_own.T, b_tilde)), x_tilde
            )
        except np.linalg.LinAlgError:
            results[key][method + "_rn_Axb"] = -1

    else:
        results[key][method + "_rn_QR"] = 0
        results[key][method + "_rn_Axb"] = 0

