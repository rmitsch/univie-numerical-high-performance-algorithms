"""
Implementation of Householder-reflection based QR decomposition of nxm matrices.
"""

import numpy as np
import algorithms as alg
import pandas as pd
import utils
import matplotlib.pyplot as plt
from tqdm import tqdm
import tests

"""
todo 
    - Implementation of up-/downdating rows/columns
    - Evaluation framework
    - Blocking
    - Comparison with performance measurements from paper(s)
    - Store in input matrix 
    - Investigate alternatives for sparse matrices/think about reasoning for ignoring them.
    - Further literature research on optimizations
    
###############################

sequence by priority:
    - delete one row
    - add one row
    - delete one col
    - add one col
    - blockify updates
    - blockify core alg
        
###############################

to ignore:
    - complex values
    
###############################

ad evaluation: measure clock time after each substiantial improvement:
    - with/without numba
    - blockify core alg
    - updating with single rs/cs
    - block updates
    
use reasonable, but smallish n, m for intermediate runs (up to 2500/2000 with inc > 100?).
use larger m, n for final evaluation run.
use Ax = b for evaluation.

###############################

Qs:
    - what to do about sparse matrices? ignore completely?

"""

# todo inner loop has to add/remove new rows/columns; update values.
# -> 5 test cases - hardcode for each case in separate function.
# versions:
#   - own
#   - scipy's QR
#   - scipiy's https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.qr_update.html
#     (or qr_insert(), qr_delete()).


if __name__ == '__main__':
    logger = utils.create_logger("logger")
    sizes = [(i + 5, i) for i in range(10, 101, 10)]
    sizes.extend([(300, 300), (400, 400), (500, 500)])

    # Compile functions with numba so that compilation time is not included in performance measurements.
    alg.compile_functions_with_numba()

    results_df = tests.test_generation_from_scratch(sizes)
    results_df.mn = np.log(results_df.mn)
    results_df.plot(x="mn", y=["time_own", "time_lib"], logy=True)

    plt.grid(True)
    plt.show()
    print(results_df[["res_norm_QR", "res_norm_Axb"]])



