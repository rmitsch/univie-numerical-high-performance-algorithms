%-------------------------------------------------
% Experiment with values for k.
%-------------------------------------------------


% Set various constants.
problem_sizes = 1500:1500:1500;
peak_performance = 3.5 * 10^9;
maxit = uint32(100);
eps = 10^-6;
num_k = 20,

runtimes = zeros(num_k, size(problem_sizes)(2));
its = zeros(num_k, size(problem_sizes)(2));

for i = 1:size(problem_sizes)(2)
	n = uint32(problem_sizes(i))

	% Generate random non-singular matrix of rank n.
    A = generate_random_symmetric_matrix(n);
    % Initialize start vector x0.
	x0 = rand(problem_sizes(i), 1);
	% Get true dominant eigenvalue.
	l = max(eig(A));

	for k = 1:1:num_k
		j_str  = num2str(k);
		tic_id = tic;
		[ ...
			lambda, ...
			v,  ...
			its(k, i), ...
			errevals.(j_str).(num2str(n)), ...
			errress.(j_str).(num2str(n)) ...
		] = rqi_k(n, A, x0, 0, eps, maxit, l, k);
		runtimes(k, i) = toc(tic_id);
	end
	
	figure('Position',[0, 0, 800, 250])
    grid on
    hold off
    
    semilogy(1:1:num_k, runtimes(:, i), 'markersize', 3, '3; Runtime;o-');
    legend ({'Runtime in seconds    '}, 'location', 'eastoutside');
    ylabel('Runtime in seconds');
    xlabel('Index of start vector (1 is random, others are canonical)');
    title (strcat('Runtimes for RQI\_k w.r.t. different k and n = ', num2str(n)), 'fontsize', 16);
end