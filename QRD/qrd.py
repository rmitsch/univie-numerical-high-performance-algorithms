"""
Implementation of Householder-reflection based QR decomposition of nxm matrices.
"""
import logging
import math
import argparse
import numpy as np
from typing import Union


def householder(alpha: float, x: np.ndarray) -> Union[np.ndarray, int]:
    """
    Computes Householder vector for alpha and x.
    :param alpha:
    :param x:
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


def qr_decomposition(A: np.ndarray, m: int, n: int, logger: logging.Logger) -> Union[np.ndarray, np.ndarray]:
    """
    Applies Householder-based QR decomposition on specified matrix A.
    :param A:
    :param m:
    :param n:
    :param logger:
    :return:
    """
    logger.info("Starting QR decomposition.")
    H = []
    R = A
    Q = A
    I = np.eye(m, m)

    for j in range(0, n):
        # Apply Householder transformation.
        x = A[j + 1:m, j]
        v_householder, tau = householder(np.linalg.norm(x), x)
        v = np.zeros((1, m))
        v[0, j] = 1
        v[0, j + 1:m] = v_householder

        # Note that this sequence of steps is not the most efficient one - see slides or
        # http://home.ku.edu.tr/~emengi/teaching/math504_f2011/Lecture12_new.pdf: Multiplying in the right sequence with
        # A directly is more performant.
        res = I - tau * v * np.transpose(v)
        # if j > 0:
            # for k in range(0, j):
            #     res[k, k] = 1

        R = np.matmul(res, R)
        H.append(res)

    R = A
    for h in H:
        print(h)
        R = np.matmul(h, R)

    return Q, R


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

    A = np.random.rand(m, n)
    q, r = np.linalg.qr(A)
    Q, R = qr_decomposition(A, m, n, logger)

    print("*****")
    print(Q)
    print(q)
    print("-----")
    print(R)
    print(r)

