import logging
import numpy as np
import scipy
import pandas as pd
from algorithms import l1 as alg1


METHODS = ["l2_scratch", "l1", "l2", "l3", "scipy_scratch", "scipy_update"]


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
        A_tilde: np.ndarray,
        Q_tilde: np.ndarray = None,
        R_tilde: np.ndarray = None,
        x_tilde: np.ndarray = None,
        b_tilde: np.ndarray = None,
        Q_tilde_own: np.ndarray = None,
        R_tilde_own: np.ndarray = None,
        compute_accuracy: bool = True
):
    """
    Adds set of measurements to result dictionary.
    :param results:
    :param method:
    :param time:
    :param A_tilde:
    :param Q_tilde:
    :param R_tilde:
    :param x_tilde:
    :param b_tilde:
    :param Q_tilde_own:
    :param R_tilde_own:
    :param compute_accuracy:
    :return:
    """
    m_tilde, n_tilde = A_tilde.shape
    key = (m_tilde, n_tilde)

    if key not in results:
        results[key] = {
            "m": m_tilde,
            "n": n_tilde,
            "mn": m_tilde * n_tilde
        }

        for m in METHODS:
            results[key][m + "_time"] = 0
            results[key][m + "_rn_QR"] = 0
            results[key][m + "_rn_Axb"] = 0

    results[key][method + "_time"] = time

    if compute_accuracy:
        print(np.allclose(np.dot(Q_tilde_own, R_tilde_own), np.dot(Q_tilde, R_tilde)))
        print(np.allclose(Q_tilde_own, Q_tilde), np.allclose(R_tilde_own, R_tilde))

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

