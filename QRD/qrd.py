"""
Implementation of Householder-reflection based QR decomposition of nxm matrices.
"""
import logging
import math
import argparse
import numpy as np
from typing import Union


def householder(alpha, x, logger: logging.Logger) -> Union[np.ndarray, int]:
    """
    Computes Householder vector for alpha and x.
    :param alpha:
    :param x:
    :param logger:
    :return:
    """

    s = math.pow(np.linalg.norm(x, ord=2), 2)
    v = x

    if s == 0:
        tau = 0
    else:
        t = math.sqrt(alpha * alpha + s)
        v_one = alpha - t if alpha <= 0 else -s / (alpha + t)

        tau = 2 * v_one * v_one / (s + v_one * v_one)
        v /= v_one

    return v, tau


def qr_decomposition(A: np.ndarray, m: int, n: int, logger: logging.Logger) -> np.ndarray:
    """
    Applies Householder-based QR decomposition on specified matrix A.
    :param A:
    :param m:
    :param n:
    :param logger:
    :return:
    """
    logger.info("Starting QR decomposition.")

    for j in range(0, n):
        print(j)

    # todo
    #   - go through individual vectors, apply householder transformation.
    #   - piece together Q^T, R, Q. compare with correct result.

    return A


if __name__ == '__main__':
    # todo
    #   generate nxm matrix
    #   timing
    #   call householder() with corresponding vectors.

    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s : %(levelname)s : %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    logger = logging.getLogger("log")

    parser = argparse.ArgumentParser(description='QR decomposition based on Householder reflections.')
    parser.add_argument('-m', '--m', dest='m', type=int, help='m for matrix A of size mxn.', required=True)
    parser.add_argument('-n', '--n', dest='n', type=int, help='n for matrix A of size mxn.', required=True)
    args = vars(parser.parse_args())
    m = args["m"]
    n = args["n"]

    A = qr_decomposition(np.random.rand(m, n), m, n, logger)

