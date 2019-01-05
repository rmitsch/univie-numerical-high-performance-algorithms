"""
Implementation of Householder-reflection based QR decomposition of nxm matrices.
"""

from algorithms import l1 as alg1
from algorithms import l2 as alg2
from algorithms import l3 as alg3
import utils
import matplotlib.pyplot as plt
import tests
import numpy as np
import pandas as pd

"""
todo 
    - Blocking
    - Comparison with performance measurements from paper(s) 
    - Investigate alternatives for sparse matrices/think about reasoning for ignoring them.
    - Further literature research on optimizations -> only in provided papers.
    - Fixing indexing bugs
    
###############################

sequence by priority:
    - blockify core alg
        
###############################

to ignore:
    - complex values
    - m <= n
    
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


def compile_functions_with_numba():
    """
    Calls all numba-decorated function with dummy data so that they are compiled lazily before evaluation.
    :return:
    """

    A = np.random.rand(10, 5)
    alg1.matmul(A, A.T)

    tests.test_del_rows(A.shape, [9])
    tests.test_add_rows(A.shape, [11])
    tests.test_del_cols(A.shape, [4])
    tests.test_add_cols(A.shape, [6])


if __name__ == '__main__':
    logger = utils.create_logger("logger")
    time_col_names = [method + "_time" for method in utils.METHODS]

    # a = np.asarray([0, 1, 2, 3])
    # print(a.shape)
    # print(a[:3])
    # print(a[2:3]) -> 2
    # print(a[2:4]) -> 2, 3
    # exit()

    # todo:
    #   - add machine-independent efficiency as metric.
    #   - last (simple) perf. improvements? e. g. ignoring d_tilde.
    #   - gather data.
    #   - presentation w/ structure and coarse content.
    #       - ideas/method behind update algos?
    #       - future work/alternatives?
    #       - how to deal w/ sparse matrices?

    # Compile functions with numba so that compilation time is not included in performance measurements.
    logger.info("Compiling numba functions.")
    # compile_functions_with_numba()

    # Always: Q.shape = (m, m); R.shape = (m, n).
    logger.info("Evaluating.")
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', None):
        # results_df = tests.test_del_rows((5, 3), [4])
        # results_df.plot(x="m", y=time_col_names, logy=True, title="Deleting rows")
        # plt.grid(True)

        # results_df = tests.test_add_rows((4, 3), [5])
        # results_df.plot(x="m", y=time_col_names, logy=True, title="Inserting rows")
        # plt.grid(True)

        # results_df = tests.test_del_cols((5, 4), [3])
        # results_df.plot(x="n", y=time_col_names, logy=True, title="Deleting columns")
        # plt.grid(True)

        results_df = tests.test_add_cols((4, 2), [3])
        # results_df.plot(x="n", y=time_col_names, logy=True, title="Inserting columns")
        # plt.grid(True)

    logger.info("Plotting results.")
    plt.show()
