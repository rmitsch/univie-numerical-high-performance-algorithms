"""
Implementation of Householder-reflection based QR decomposition of nxm matrices.
"""

import numpy as np
import algorithms as alg
import time
import pandas as pd
import utils
import matplotlib.pyplot as plt
from tqdm import tqdm
import scipy
import scipy.linalg._decomp_update as scipy_qr_update

"""
todo 
- Implementation of up-/downdating rows/columns
- Evaluation framework
- Blocking
- Parallelization of Householder transformations
- Numba
- Comparison with performance measurements from paper(s)
- Further literature research on optimizations
"""


if __name__ == '__main__':
    logger = utils.create_logger("logger")
    sizes = [(i, i) for i in range(10, 101, 10)]
    sizes.extend([(300, 300), (400, 400)])

    results = {
        "m": [],
        "n": [],
        "mn": [],
        "time_own": [],
        "time_lib": [],
        "res_norm": []
    }

    # Compile functions with numba so that compilation time is not included in performance measurements.
    alg.compile_functions_with_numba()

    pbar = tqdm(total=np.sum(m * n for m, n in sizes))
    for size in sizes:
        m, n = size
        A = np.random.rand(m, n)

        # todo inner loop has to add/remove new rows/columns; update values.
        # -> 5 test cases - hardcode for each case in separate function.
        # blocking is separate execution calling unblocked versions of functions. versions:
        #   - own
        #   - scipy's QR
        #   - scipiy's https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.qr_update.html
        #     (or qr_insert(), qr_delete()).

        start = time.time()
        scipy.linalg.qr(A)
        duration_lib = time.time() - start

        start = time.time()
        Q, R = alg.qr_decomposition(A)
        duration_own = time.time() - start

        results["time_own"].append(duration_own)
        results["time_lib"].append(duration_lib)
        results["m"].append(m)
        results["n"].append(n)
        results["mn"].append(m * n)
        results["res_norm"].append(alg.compute_residual(alg.matmul(Q, R), A))

        pbar.update(m * n)
    pbar.close()

    results_df = pd.DataFrame(results)
    results_df.mn = np.log(results_df.mn)
    results_df.plot(x="mn", y=["time_own", "time_lib"], logy=True)

    plt.grid(True)
    plt.show()
    print(results_df)



