#####################################
# Run LU decomposition & evaluation.
#####################################


problem_sizes = 100:100:500;
rel_residuals_unblocked = zeros(size(problem_sizes));
rel_fw_errors_unblocked = zeros(size(problem_sizes));
rel_factorization_errors_unblocked = zeros(size(problem_sizes));
efficiency_unblocked = zeros(size(problem_sizes));
runtimes_unblocked = zeros(size(problem_sizes));
rel_residuals_blocked = zeros(size(problem_sizes));
rel_fw_errors_blocked = zeros(size(problem_sizes));
rel_factorization_errors_blocked = zeros(size(problem_sizes));
efficiency_blocked = zeros(size(problem_sizes));
runtimes_blocked = zeros(size(problem_sizes));
peak_performance = 3.29 * 10^9; # https://asteroidsathome.net/boinc/cpu_list.php

for i = 1:size(problem_sizes)(2)
    printf('n = %i\n', problem_sizes(i));

    # Generate random non-singular matrix of rank n.
    A = generate_random_nonsingular_matrix(problem_sizes(i));

    # Execute unblocked LU decomposition.
    [ ...
        rel_residuals_unblocked(i),  ...
        rel_fw_errors_unblocked(i),  ...
        rel_factorization_errors_unblocked(i),  ...
        runtimes_unblocked(i) ...
    ] = pluStats(A, problem_sizes(i));
    efficiency_unblocked(i) = 2 * (problem_sizes(i)^3) / 3 / runtimes_unblocked(i) / peak_performance; 
    
    # Execute blocked LU decomposition.
    [ ...
        rel_residuals_blocked(i),  ...
        rel_fw_errors_blocked(i),  ...
        rel_factorization_errors_blocked(i),  ...
        runtimes_blocked(i) ...
    ] = upluStats(A, problem_sizes(i));
    efficiency_blocked(i) = 2 * (problem_sizes(i)^3 / 3) / runtimes_blocked(i) / peak_performance;
end

plot_results(
    rel_residuals_unblocked, 
    rel_fw_errors_unblocked, 
    rel_factorization_errors_unblocked, 
    runtimes_unblocked,
    efficiency_unblocked, 
    problem_sizes,
    " unblocked"
)

plot_results(
    rel_residuals_blocked, 
    rel_fw_errors_blocked, 
    rel_factorization_errors_blocked, 
    runtimes_blocked, 
    efficiency_blocked,
    problem_sizes,
    " blocked"
)