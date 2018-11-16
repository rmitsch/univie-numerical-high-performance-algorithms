% todos
% o	1.  finish invit.m with all results.
% o 5.  calculate efficiencies.
% o	6.  write plotting code.
% o 7.  choose starting vectors (what are canonical starting vectors?).
% o	8.  conduct experiments with shift for invit.
% o	9.  conduct experiments with k for rqi_k.
% o	11. documentation

% Set various constants.
problem_sizes = 10:10:10;
k = 3;
peak_performance = 3.29 * 10^9; % https://asteroidsathome.net/boinc/cpu_list.php
maxit = uint32(1000);
eps = 10^-4;
eig_methods = {'invit', 'rqi', 'rqi_k'};
sigma = -32;

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
	n = uint32(problem_sizes(i));

	% Generate random non-singular matrix of rank n.
    A = generate_random_symmetric_matrix(n);
    % Initialize start vector x0.
	x0 = rand(problem_sizes(i), 1);
	% Get true dominant eigenvalue.
	l = max(eig(A));

	%A = [3, 1; 1, 3];
	%x0 = [0; 1];
	%n = 2;
	

	for method = eig_methods
		method = method{1};
		
		if (strcmp(method, 'rqi'))
			method
			tic_id = tic;
			
			% Suppress warnings about nearly singular matrix.
			warning('off', 'Octave:nearly-singular-matrix');

			% Note that k is ignored in invit() and rqi(), while sigma is ignored in rqi() and rqi_k().
			[ ...
				lambda, ...
				v,  ...
				its.(method)(i), ...
				errevals.(method).(num2str(i)), ...
				errress.(method).(num2str(i)) ...
			] = funcs.(method)(n, A, x0, sigma, eps, maxit, l, k);
			it = its.(method)(i)
			lambda
			l
			v;
			runtimes.(method)(i) = toc(tic_id);
			complexity = 2 * (problem_sizes(i)^3) / 3;
			efficiencies.(method)(i) = complexity / runtimes.(method)(i) / peak_performance;
 
		endif
	end
end

%%%%%%%%%%%%%%%%%%%%
% 		Plots	   %
%%%%%%%%%%%%%%%%%%%%

% over all problem sizes; one series per method; problem size on x-axis:
%	- (1) absolute runtimes
%	- (2) number of iterations until convergence
%	- (3) avg. runtime per iteration
%	- (4) efficiency

% (5) convergence history

itis = 'done'