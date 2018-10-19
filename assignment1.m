#####################################
# Run LU decomposition & evaluation.
#####################################


problem_sizes = 100:100:1200;
rel_residuals_unblocked = zeros(size(problem_sizes));
rel_fw_errors_unblocked = zeros(size(problem_sizes));
rel_factorization_errors_unblocked = zeros(size(problem_sizes));
runtimes_unblocked = zeros(size(problem_sizes));
rel_residuals_blocked = zeros(size(problem_sizes));
rel_fw_errors_blocked = zeros(size(problem_sizes));
rel_factorization_errors_blocked = zeros(size(problem_sizes));
runtimes_blocked = zeros(size(problem_sizes));

for i = 1:size(problem_sizes)(2)
    printf('n = %i\n', problem_sizes(i));

    # Generate random non-singular matrix of rank n.
    A = generate_random_nonsingular_matrix(problem_sizes(i));

    
    #{
    # Execute unblocked LU decomposition.
    [ ...
        rel_residuals_unblocked(i),  ...
        rel_fw_errors_unblocked(i),  ...
        rel_factorization_errors_unblocked(i),  ...
        runtimes_unblocked(i) ...
    ] = pluStats(A, problem_sizes(i));
    #}
    
    # Execute blocked LU decomposition.
    [ ...
        rel_residuals_blocked(i),  ...
        rel_fw_errors_blocked(i),  ...
        rel_factorization_errors_blocked(i),  ...
        runtimes_blocked(i) ...
    ] = upluStats(A, problem_sizes(i));
    
end
runtimes_blocked
#runtimes_unblocked
#{
plot_results(
    rel_residuals_unblocked, 
    rel_fw_errors_unblocked, 
    rel_factorization_errors_unblocked, 
    runtimes_unblocked, 
    problem_sizes,
    " unblocked"
)


plot_results(
    rel_residuals_blocked, 
    rel_fw_errors_blocked, 
    rel_factorization_errors_blocked, 
    runtimes_blocked, 
    problem_sizes,
    " blocked"
)
#}