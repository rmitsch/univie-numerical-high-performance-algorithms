% todos
% EVALUTION
% 	o choose sigma for comparison of invit/rqi/rqi_k in assignment_2.m.
% 	o documentation
% EXPERIMENTS
% 	o k for rqi_k.
%	o case of RQI advantage/disadvantage. note: unattractive when factorization is expensive. + reasoning.

% Set various constants.
problem_sizes = 3:3:3;
k = 3;
peak_performance = 3.5 * 10^9; % https://lhcathome.cern.ch/lhcathome/cpu_list.php; theoretical peak performance not found.
maxit = uint32(1000);
eps = 10^-6;
eig_methods = {'invit', 'rqi', 'rqi_k'};
sigma = -4;

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
	l = max(eig(A));
	
	
	A = [3.90439, 0.70473, 0.63478; 0.70473, 3.08147, 0.12270; 0.63478, 0.12270, 3.80788];
	x0 = [0.48914; 0.75587; 0.67534];
	n = uint32(3);
	l = max(eig(A))
	eig(A)
	

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
		eig(A)
		runtimes.(method)(i) = toc(tic_id);
		
		num_ops = calc_num_ops(method, n, its.(method)(i), k);
		efficiencies.(method)(i) = num_ops / runtimes.(method)(i) / peak_performance;
		exit
	end
end

%%%%%%%%%%%%%%%%%%%%
% 		Plots	   %
%%%%%%%%%%%%%%%%%%%%

plot_results(problem_sizes, runtimes, its, efficiencies, errevals, errress);