# todos
#	1. finish invit.m with all results.
#	2. implement rqi and rqi_k.
#	3. generate matrices without complex eigenvalues.
#	4. use LU factorization instead of direct solving in routines.
#	5. write plotting code.
#	6. conduct experiments with shift for invit.
#	7. conduct experiments with k for rqi_k.
#	8. check input for all routines
#	9. documentation

# Set various constants.
problem_sizes = 5:5:5;
k = 3;
peak_performance = 3.29 * 10^9; # https://asteroidsathome.net/boinc/cpu_list.php
maxit = 10;
eps = 10^-6;
eig_methods = {'invit', 'rqi', 'rqi_k'};
sigma = 32;

# Set function handles.
funcs.invit = @invit;
funcs.rqi = @rqi;
funcs.rqi_k = @rqi_k;

# Order of methods in 2D arrays: invit, rqi, rqi_k.
for method = eig_methods
	method = method{1};
	# Note: Accuracy metrics are stored in errevals.(method).(i) and errress.(method).(i), respectively.
	
	# Efficiency metrics.
	efficiencies.(method) = zeros(size(problem_sizes));
	runtimes.(method) = zeros(size(problem_sizes));
	its.(method) = zeros(size(problem_sizes));
end

for i = 1:size(problem_sizes)(2)
	n = problem_sizes(i)

	# Generate random non-singular matrix of rank n.
    A = rand(problem_sizes(i));
    # Initialize random vector x0.
	x0 = rand(problem_sizes(i), 1);


	A = [3, 1; 1, 3];
	x0 = [0; 1];
	n = 2;
	l = max(eig(A));

	for method = eig_methods
		method = method{1};
		
		if (strcmp(method, 'rqi'))
			method
			tic_id = tic;
			# Note that k is ignored in invit() and rqi().
			[ ...
				lambda, ...
				v,  ...
				its.(method)(i), ...
				errevals.(method).(num2str(i)), ...
				errress.(method).(num2str(i)) ...
			] = funcs.(method)(n, A, x0, sigma, eps, maxit, l, k);
			
			lambda
			l
			v
			runtimes.(method)(i) = toc(tic_id);
			# todo - calculate efficiencies
			efficiencies.(method)(i) = toc(tic_id);
		endif
	end

end

itis = 'done'