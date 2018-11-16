% todos
% BUGS
% 	o finish invit.m with all results.
% EVALUTION
% 	o calculate efficiencies.
%		-> what's the runtime complexity? check pdf
% 	o choose starting vectors (what are canonical starting vectors?).
% 	o documentation
% EXPERIMENTS
% 	o shift for invit.
% 	o k for rqi_k.
%	o case of RQI advantage/disadvantage. note: unattractive when factorization is expensive. + reasoning.

% Set various constants.
problem_sizes = 100:100:300;
k = 3;
peak_performance = 3.29 * 10^9; % https://asteroidsathome.net/boinc/cpu_list.php
maxit = uint32(100);
eps = 10^-6;
eig_methods = {'invit', 'rqi', 'rqi_k'};
sigma = 0;

% Set function handles.
funcs.invit = @invit;
funcs.rqi = @rqi;
funcs.rqi_k = @rqi_k;

% Order of methods in 2D arrays: invit, rqi, rqi_k.
for method = eig_methods
	method = method{1};
	% Note: Accuracy metrics are stored in errevals.(method).(i) and errress.(method).(i), respectively.
	
	% Efficiency metrics.
	efficiencies.(method) = zeros(size(problem_sizes));
	runtimes.(method) = zeros(size(problem_sizes));
	its.(method) = zeros(size(problem_sizes));
end

for i = 1:size(problem_sizes)(2)
	n = uint32(problem_sizes(i))

	% Generate random non-singular matrix of rank n.
    A = generate_random_symmetric_matrix(n);
    % Initialize start vector x0.
	x0 = rand(problem_sizes(i), 1);
	% Get true dominant eigenvalue.
	l = max(eig(A))
	
	for method = eig_methods
		method = method{1};
		printf('  %s\n', method)
		tic_id = tic;
		
		% Suppress warnings about nearly singular matrix.
		warning('off', 'Octave:nearly-singular-matrix');

		% Note that k is ignored in invit() and rqi(), while sigma is ignored in rqi() and rqi_k().
		[ ...
			lambda, ...
			v,  ...
			its.(method)(i), ...
			errevals.(method).(num2str(n)), ...
			errress.(method).(num2str(n)) ...
		] = funcs.(method)(n, A, x0, sigma, eps, maxit, l, k);
		
		lambda

		runtimes.(method)(i) = toc(tic_id);
		complexity = 2 * (problem_sizes(i)^3) / 3;
		efficiencies.(method)(i) = complexity / runtimes.(method)(i) / peak_performance;
	end
	printf('\n-----------------\n')
end

%%%%%%%%%%%%%%%%%%%%
% 		Plots	   %
%%%%%%%%%%%%%%%%%%%%

plot_results(problem_sizes, runtimes, its, efficiencies, errevals, errress);