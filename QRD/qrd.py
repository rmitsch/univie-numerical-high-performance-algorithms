"""
Implementation of Householder-reflection based QR decomposition of nxm matrices.
"""
import math
import argparse
import numpy as np


def householder(alpha, x):
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


if __name__ == '__main__':
    # todo
    #   generate nxm matrix
    #   call householder() with corresponding vectors.

    parser = argparse.ArgumentParser(description='QR decomposition based on Householder reflections.')
    parser.add_argument('-m', '--m', dest='m', type=int, help='m for matrix A of size mxn.', required=True)
    parser.add_argument('-n', '--n', dest='n', type=int, help='n for matrix A of size mxn.', required=True)
    args = vars(parser.parse_args())

    print(args["n"])
