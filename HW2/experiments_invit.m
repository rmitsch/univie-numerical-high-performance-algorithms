%-------------------------------------------------
% Experiment with shift values.
%-------------------------------------------------

% Set various constants.
problem_sizes = 500:500:500;
peak_performance = 3.5 * 10^9;
maxit = uint32(1000);
eps = 10^-6;

for i = 1:size(problem_sizes)(2)
	n = uint32(problem_sizes(i))

	% Generate random non-singular matrix of rank n.
    A = generate_random_symmetric_matrix(n);
    % Initialize start vector x0.
	x0 = rand(problem_sizes(i), 1);
	% Get true dominant eigenvalue.
	l = max(eig(A));
	
	shift_distance = l * 0.25;
	shift_interval = shift_distance * 0.1;
	n_shifts = shift_distance / shift_interval + 1;

	runtimes = zeros(n_shifts, 1);
	its = zeros(n_shifts, 1);

	sigmas = [];
	for j = 1:1:n_shifts
		sigma = -shift_distance / 2 + (j - 1) * shift_interval + l;
		sigmas = [sigmas sigma];
		eig_dist_i_str = num2str(j);
		 
		tic_id = tic;
		[ ...
			lambda, ...
			v,  ...
			its(j), ...
			errevals.(eig_dist_i_str).(num2str(n)), ...
			errress.(eig_dist_i_str).(num2str(n)) ...
		] = invit(n, A, x0, sigma, eps, maxit, l);
		
		runtimes(j) = toc(tic_id);
	end
	
	figure('Position',[0, 0, 800, 250])
    grid on
    hold off
    semilogy(sigmas, runtimes, 'markersize', 3, '3; Runtime;o-');
    legend ({'Runtime in seconds    '}, 'location', 'eastoutside');
    ylabel('Runtime in seconds');
    xlabel('abs(sigma - max(eig(A))');
    title (strcat('Runtimes w.r.t. different shift values for n = ', num2str(n)), 'fontsize', 16);
end
exit

%-------------------------------------------------
% Experiment with starting vectors..
%-------------------------------------------------


% Set various constants.
problem_sizes = 100:100:500;
peak_performance = 3.5 * 10^9;
maxit = uint32(100);
eps = 10^-6;
num_start_vectors = 4 + 1;

runtimes = zeros(num_start_vectors, size(problem_sizes)(2));
its = zeros(num_start_vectors, size(problem_sizes)(2));

for i = 1:size(problem_sizes)(2)
	n = uint32(problem_sizes(i))

	% Generate random non-singular matrix of rank n.
    A = generate_random_symmetric_matrix(n);
    sigma = max(eig(A)) * 0.9;
    % Initialize start vector x0.
	x0 = rand(problem_sizes(i), 1);
	% Get true dominant eigenvalue.
	l = max(eig(A));
	
	start_vectors = [];
	start_vectors = [start_vectors, rand(n, 1)];
	for j = 2:1:num_start_vectors
		start_vector = zeros(n, 1);
		start_vector(uint32(rand * n)) = 1;
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
		] = invit(n, A, x0, sigma, eps, maxit, l);
		l, lambda
		runtimes(j, i) = toc(tic_id);
	end
	
	figure('Position',[0, 0, 800, 250])
    grid on
    hold off
    
    semilogy(1:1:num_start_vectors, runtimes(:, i), 'markersize', 3, '3; Runtime;o-');
    legend ({'Runtime in seconds    '}, 'location', 'eastoutside');
    ylabel('Runtime in seconds');
    xlabel('Index of start vector (1 is random)');
    title (strcat('Runtimes w.r.t. different start vectors for n = ', num2str(n)), 'fontsize', 16);
end

%errevals
%errress
%efficiencies
%runtimes
%its