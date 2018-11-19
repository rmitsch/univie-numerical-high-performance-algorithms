%-------------------------------------------------
% Experiment with starting vectors.
%-------------------------------------------------


% Set various constants.
problem_sizes = 1000:1000:1000;
peak_performance = 3.5 * 10^9;
maxit = uint32(100);
eps = 10^-6;
num_start_vectors = 10 + 1;

runtimes = zeros(num_start_vectors, size(problem_sizes)(2));
its = zeros(num_start_vectors, size(problem_sizes)(2));

for i = 1:size(problem_sizes)(2)
	n = uint32(problem_sizes(i))

	% Generate random non-singular matrix of rank n.
    A = generate_random_symmetric_matrix(n);
	% Get true dominant eigenvalue.
	l = max(eig(A));
	
	start_vectors = [];
	start_vectors = [start_vectors, rand(n, 1)];
	for j = 2:1:num_start_vectors
		start_vector = zeros(n, 1);
		start_vector(ceil(rand * n)) = 1;
		start_vectors = [start_vectors, start_vector]; 
	end

	for j = 1:1:num_start_vectors
		j_str  = num2str(j);
		tic_id = tic;
		[ ...
			lambda, ...
			v,  ...
			its(j, i), ...
			errevals.(j_str).(num2str(n)), ...
			errress.(j_str).(num2str(n)) ...
		] = rqi(n, A, start_vectors(:, j), 0, eps, maxit, l);
		runtimes(j, i) = toc(tic_id);
	end
	
	figure('Position',[0, 0, 800, 250])
    grid on
    hold off
    
    semilogy(1:1:num_start_vectors, runtimes(:, i), 'markersize', 3, '3; Runtime;o-');
    legend ({'Runtime in seconds    '}, 'location', 'eastoutside');
    ylabel('Runtime in seconds');
    xlabel('Index of start vector (1 is random, others are canonical)');
    title (strcat('Runtimes for RQI w.r.t. different start vectors for n = ', num2str(n)), 'fontsize', 16);
end


%-------------------------------------------------
% Show case of advantage of RQI.
%-------------------------------------------------


% Set various constants.
problem_sizes = 200:200:1000;
maxit = uint32(100);
eps = 10^-6;
num_start_vectors = 10 + 1;

runtimes = zeros(2, size(problem_sizes)(2));
its = zeros(2, size(problem_sizes)(2));

warning('off', 'Octave:nearly-singular-matrix');
for i = 1:size(problem_sizes)(2)
	n = uint32(problem_sizes(i))

	% Generate random non-singular matrix of rank n.
	A = rand(n, n);
	A = tril(A);
	x0 = rand(problem_sizes(i), 1);
    % Get true dominant eigenvalue.
	l = max(eig(A));
	
	tic_id = tic;
	[lambda, v, its(1, i), erreval, errress] = rqi(n, A, x0, 0, eps, maxit, l);
	runtimes(1, i) = toc(tic_id);

	tic_id = tic;
	[lambda, v, its(2, i), erreval, errress] = invit(n, A, x0, 0, eps, maxit, l);
	runtimes(2, i) = toc(tic_id);
	
end

figure('Position',[0, 0, 800, 250])
grid on
hold on

semilogy(1:1:size(problem_sizes)(2), runtimes(1, :), 'markersize', 3, '3; Runtime;o-');
semilogy(1:1:size(problem_sizes)(2), runtimes(2, :), 'marker', 'x',  'markersize', 8, '1; Inverse Iteration;--');	

leg = legend ({
        'Runtime for invit()', 
        'Runtime for rqi()'
    }, 'location', 'eastoutside');
set (leg, 'fontsize', 16);
title ('Runtimes with lower triangular matrix', 'fontsize', 16);
ylabel('Accuracy metric value');
xlabel('Iteration number');

figure('Position',[0, 0, 800, 250])
grid on
hold on

semilogy(1:1:size(problem_sizes)(2), its(1, :), 'markersize', 3, '3; Runtime;o-');
semilogy(1:1:size(problem_sizes)(2), its(2, :), 'marker', 'x',  'markersize', 8, '1; Inverse Iteration;--');	

leg = legend ({
        'Number of iterations for invit()', 
        'Number of iterations for rqi()'
    }, 'location', 'eastoutside');
set (leg, 'fontsize', 16);
title ('Number of iterations with lower triangular matrix', 'fontsize', 16);
ylabel('Accuracy metric value');
xlabel('Iteration number');