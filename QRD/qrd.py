"""
Implementation of Householder-reflection based QR decomposition of nxm matrices.
"""
import logging

import argparse
import numpy as np
from typing import Union


def householder(x: np.ndarray) -> Union[np.ndarray, int]:
    """
    Computes v and tau for Householder transformation of x.
    Note that
    https://stackoverflow.com/questions/53489237/qr-decomposition-with-householder-transformations-introduction-of-zeroes-uncle
    helped in getting the implementation right.
    :param x: Vector.
    :return:
    """

    # Use vectorized approach as shown in https://rosettacode.org/wiki/QR_decomposition#Python.
    v = x / (x[0] + np.copysign(np.linalg.norm(x), x[0]))
    v[0] = 1
    tau = 2 / (v.T @ v)

    # Down here: Approach as in paper with slight modifications.
    # alpha = x[0]
    # s = np.power(np.linalg.norm(x[1:]), 2)
    # v = x.copy()
    #
    # if s == 0:
    #     tau = 0
    # else:
    #     t = np.sqrt(alpha**2 + s)
    #     v[0] = alpha - t if alpha <= 0 else -s / (alpha + t)
    #
    #     tau = 2 * v[0]**2 / (s + v[0]**2)
    #     v /= v[0]

    return v, tau


def qr_decomposition(A: np.ndarray, logger: logging.Logger) -> Union[np.ndarray, np.ndarray]:
    """
    Decomposes rectangular matrix A in matrices Q and R.
    Note that
    https://stackoverflow.com/questions/53489237/qr-decomposition-with-householder-transformations-introduction-of-zeroes-uncle
    helped in getting the implementation right.
    :param A:
    :param logger:
    :return:
    """
    logger.info("Starting QR decomposition.")

    m, n = A.shape
    R = A.copy()
    Q = np.identity(m)

    for j in range(0, n):
        # Apply Householder transformation.
        v, tau = householder(R[j:, j])
        H = np.identity(m)
        H[j:, j:] -= tau * v.reshape(-1, 1) * v
        R = H @ R
        Q = H @ Q

    # Note that triu is used here mainly for cosmetic reasons - we discard of the entries under the diagonale.
    return Q[:n].T, np.triu(R[:n])


if __name__ == '__main__':
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
    Q, R = qr_decomposition(A, logger)

    print("*****")
    print(Q)
    print(q)
    print("-----")
    print(R)
    print(r)

