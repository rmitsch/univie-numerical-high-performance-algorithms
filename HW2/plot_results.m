function plot_results(n, runtimes, its, efficiencies, errevals, errress)

	% ----------------------------
	% Plot absolute runtimes.
	% ----------------------------

	figure('Position',[0, 0, 800, 250])
	grid on
	hold on

	semilogy(n, runtimes.invit, 'marker', 'x',  'markersize', 8, '1; Inverse Iteration;.-');	
	semilogy(n, runtimes.rqi, 'marker', '+',  'markersize', 8, '2; Rayleigh Quotient Iteration;o-');
	semilogy(n, runtimes.rqi_k, 'marker', 's',  'markersize', 8, '3; Rayleigh Quotient Iteration (k);.-');

	leg = legend ({
	        'Inverse Iteration', 
	        'Rayleigh Quotient Iteration', 
	        'Rayleigh Quotient Iteration (k)'
	    }, 'location', 'eastoutside');
	set (leg, 'fontsize', 16);
	title ('Runtime w.r.t. problem size', 'fontsize', 16);
	ylabel('Runtime (sec)');
	xlabel('Problem size');

	% ----------------------------	
	% Plot avg. runtime per iteration.
	% ----------------------------

	figure('Position',[0, 0, 800, 250])
	grid on
	hold on

	semilogy(n, runtimes.invit ./ its.invit, 'marker', 'x',  'markersize', 8, '1; Inverse Iteration;.-');	
	semilogy(n, runtimes.rqi ./ its.rqi, 'marker', '+',  'markersize', 8, '2; Rayleigh Quotient Iteration;o-');
	semilogy(n, runtimes.rqi_k ./ its.rqi_k, 'marker', 's',  'markersize', 8, '3; Rayleigh Quotient Iteration (k);.-');

	leg = legend ({
	        'Inverse Iteration', 
	        'Rayleigh Quotient Iteration', 
	        'Rayleigh Quotient Iteration (k)'
	    }, 'location', 'eastoutside');
	set (leg, 'fontsize', 16);
	title ('Avg. runtime per iteration w.r.t. problem size', 'fontsize', 16);
	ylabel('Runtime per iteration (sec)');
	xlabel('Problem size');

	% ----------------------------
	% Plot number of iterations until convergence.
	% ----------------------------

	figure('Position',[0, 0, 800, 250])
	grid on
	hold on

	semilogy(n, its.invit, 'marker', 'x',  'markersize', 8, '1; Inverse Iteration;.-');	
	semilogy(n, its.rqi, 'marker', '+',  'markersize', 8, '2; Rayleigh Quotient Iteration;o-');
	semilogy(n, its.rqi_k, 'marker', 's',  'markersize', 8, '3; Rayleigh Quotient Iteration (k);.-');

	leg = legend ({
	        'Inverse Iteration', 
	        'Rayleigh Quotient Iteration', 
	        'Rayleigh Quotient Iteration (k)'
	    }, 'location', 'eastoutside');
	set (leg, 'fontsize', 16);
	title ('Number of iterations until convergence w.r.t. problem size', 'fontsize', 16);
	ylabel('Number of iterations');
	xlabel('Problem size');

	% ----------------------------
	% Plot machine-independent efficiency.
	% ----------------------------

	figure('Position',[0, 0, 800, 250])
	grid on
	hold on

	plot(n, efficiencies.invit, 'marker', 'x',  'markersize', 8, '1; Inverse Iteration;.-');	
	plot(n, efficiencies.rqi, 'marker', '+',  'markersize', 8, '2; Rayleigh Quotient Iteration;o-');
	plot(n, efficiencies.rqi_k, 'marker', 's',  'markersize', 8, '3; Rayleigh Quotient Iteration (k);.-');

	leg = legend ({
	        'Inverse Iteration', 
	        'Rayleigh Quotient Iteration', 
	        'Rayleigh Quotient Iteration (k)'
	    }, 'location', 'eastoutside');
	set (leg, 'fontsize', 16);
	title ('Machine-independent efficiency w.r.t. problem size', 'fontsize', 16);
	ylabel('Efficiency as percentage');
	xlabel('Problem size');

	% ----------------------------
	% Accuracy metrics.
	% ----------------------------

	idx = 1;
%	for i = n
	idx = size(n)(2);
	i = num2str(n(idx));
	
	figure('Position',[0, 0, 800, 250])
	grid on
	hold on

	semilogy(1:1:its.invit(idx), errevals.invit.(i), 'marker', 'x',  'markersize', 8, '1; Inverse Iteration;-');	
	semilogy(1:1:its.rqi(idx), errevals.rqi.(i), 'marker', '+',  'markersize', 8, '2; Rayleigh Quotient Iteration;-');
	semilogy(1:1:its.rqi_k(idx), errevals.rqi_k.(i), 'marker', 's',  'markersize', 8, '3; Rayleigh Quotient Iteration (k);-');

	semilogy(1:1:its.invit(idx), errress.invit.(i), 'marker', 'x',  'markersize', 8, '1; Inverse Iteration;--');	
	semilogy(1:1:its.rqi(idx), errress.rqi.(i), 'marker', '+',  'markersize', 8, '2; Rayleigh Quotient Iteration;--');
	semilogy(1:1:its.rqi_k(idx), errress.rqi_k.(i), 'marker', 's',  'markersize', 8, '3; Rayleigh Quotient Iteration (k);--');

	leg = legend ({
	        'Relative eigenvalue error for invit()', 
	        'Relative eigenvalue error for rqi()', 
	        'Relative eigenvalue error for rqi\_k()',
	        'Residual vector norm for invit()', 
	        'Residual vector norm for rqi()', 
	        'Residual vector norm for rqi\_k()'
	    }, 'location', 'eastoutside');
	set (leg, 'fontsize', 16);
	title (strcat('Convergence history via accuracy metrics over iterations with n = ', i), 'fontsize', 16);
	ylabel('Accuracy metric value');
	xlabel('Iteration number');

		idx += 1;
%	end
end