"""
Implementation of Householder-reflection based QR decomposition of nxm matrices.
"""

import numpy as np
import logging
import algorithms as alg
import logging
import time
import numba
import utils


if __name__ == '__main__':
    logger = utils.create_logger("logger")
    sizes = [(5, 5)]
    results = {
        "m": [],
        "n": [],
        "time": [],
        "res_norm": []
    }

    # Compile functions with numba so that compilation time is not included in performance measurements.
    alg.compile_functions_with_numba()

    for size in sizes:
        m, n = size
        A = np.random.rand(m, n)

        start = time.time()
        # Q, R = algorithms.qr_decomposition(A, logger)
        q, r = np.linalg.qr(A)
        duration = time.time() - start

        results["time"].append(duration)
        results["m"].append(m)
        results["n"].append(n)
        results["res_norm"].append(alg.compute_residual(alg.matmul(q, r), A))


