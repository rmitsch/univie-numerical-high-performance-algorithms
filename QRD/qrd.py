"""
Implementation of Householder-reflection based QR decomposition of nxm matrices.
"""

from algorithms import l1 as alg
import utils
import matplotlib.pyplot as plt
import tests
import numpy as np
import pandas as pd

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
    time_col_names = [method + "_time" for method in utils.METHODS]
    # a = np.asarray([0, 1, 2, 3])
    # print(a.shape)
    # print(a[:3])
    # print(a[2:3]) -> 2
    # print(a[2:4]) -> 2, 3
    # exit()

    # Compile functions with numba so that compilation time is not included in performance measurements.
    alg.compile_functions_with_numba()

    # sizes = [(i + 5, i) for i in range(10, 101, 10)]
    # sizes.extend([(300, 300), (400, 400), (500, 500)])
    # results_df = tests.test_generation_from_scratch(sizes)
    # results_df = results_df.drop(["l1", "l3"])
    # results_df.plot(x="mn", y=["time_own", "time_lib"], logy=True)

    # Always: Q.shape = (m, m); R.shape = (m, n).
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', None):
        # results_df = tests.test_del_rows((10, 5), [9, 8, 7])
        # results_df.plot(x="m", y=time_col_names, logy=True, title="Deleting rows")
        # plt.grid(True)
        # print(results_df)

        results_df = tests.test_add_rows((10, 5), [11, 12, 13])
        results_df.plot(x="m", y=time_col_names, logy=True, title="Inserting rows")
        plt.grid(True)
        print(results_df)

        # results_df = tests.test_del_cols((10, 5), [4, 3, 2])
        # results_df.plot(x="n", y=time_col_names, logy=True, title="Deleting columns")
        # plt.grid(True)
        # print(results_df)
        #
        # results_df = tests.test_add_cols((10, 5), [6, 7, 8])
        # results_df.plot(x="n", y=time_col_names, logy=True, title="Inserting columns")
        # plt.grid(True)
        # print(results_df)

    plt.show()
