problem_sizes = 5:5:10;
k = 3;
peak_performance = 3.29 * 10^9; # https://asteroidsathome.net/boinc/cpu_list.php

# Order of methods in 2D arrays: invit, rqi, rqi_k.
for method = {'invit', 'rqi', 'rqi_k'}
	# Accuracy metrics.
	rel_eig_error.(method{1}) = zeros(size(problem_sizes));
	res_norm.(method{1}) = zeros(size(problem_sizes));

	# Efficiency metrics.
	efficiencies.(method{1}) = zeros(size(problem_sizes));
	runtimes.(method{1}) = zeros(size(problem_sizes));
	runtimes_per_iter.(method{1}) = zeros(size(problem_sizes));
	num_iters.(method{1}) = zeros(size(problem_sizes));
	convergence_history.(method{1}) = zeros(size(problem_sizes));
end

for i = 1:size(problem_sizes)(2)
	printf('n = %i\n', problem_sizes(i));

	# Generate random non-singular matrix of rank n.
    A = rand(problem_sizes(i));
    # Initialize random vector x0.
	x0 = rand(1, problem_sizes(i));

	#[lambda, v, it, erreval, errres] = invit(n, A, x0, sigma, eps, maxit, l);

	#[lambda, v, it, erreval, errres] = rqi(n, A, x0, sigma, eps, maxit, l);

	#[lambda, v, it, erreval, errres] = rqi_k(n, A, x0, sigma, eps, maxit, l);

end
