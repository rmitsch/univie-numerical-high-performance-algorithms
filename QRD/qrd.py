"""
Implementation of Householder-reflection based QR decomposition of nxm matrices.
"""

from algorithms import l1 as alg
import utils
import matplotlib.pyplot as plt
import tests
import numpy as np

"""
todo 
    - Implementation of up-/downdating rows/columns
    - Evaluation framework
    - Blocking
    - Comparison with performance measurements from paper(s)
    - Store in input matrix 
    - Investigate alternatives for sparse matrices/think about reasoning for ignoring them.
    - Further literature research on optimizations -> only in provided papers.
    
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
    - updating with single rs/cs
    - block updates
    
use reasonable, but smallish n, m for intermediate runs (up to 2500/2000 with inc > 100?).
use larger m, n for final evaluation run.
use Ax = b for evaluation.

###############################

Qs:
    - what to do about sparse matrices? ignore completely?

"""

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

    # q: m, m
    # r: m, n

    # a = np.asarray([0, 1, 2, 3])
    # print(a.shape)
    # print(a[:3])
    # print(a[2:3]) -> 2
    # print(a[2:4]) -> 2, 3
    # exit()

    # Compile functions with numba so that compilation time is not included in performance measurements.
    alg.compile_functions_with_numba()

    # results_df = tests.test_generation_from_scratch(sizes)
    # results_df.mn = np.log(results_df.mn)
    # results_df.plot(x="mn", y=["time_own", "time_lib"], logy=True)

    # results_df = tests.test_del_row((10, 5), [(6, 5)])
    results_df = tests.test_add_row((10, 5), [(12, 5)])
    # results_df = tests.test_del_col((10, 5), [(10, 4)])
    # results_df = tests.test_add_col((10, 5), [(10, 6)])

    plt.grid(True)
    plt.show()
    print(results_df[["res_norm_QR", "res_norm_Axb"]])



