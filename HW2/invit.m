% Inverse iteration with shift.
%
% Input:
% - n: dimension (scalar)
% - A: n × n matrix
% - x0: starting vector of size n
% - sigma: shift/eigenvalue approximation (scalar)
% - eps: error tolerance (scalar)
% - maxit: the maximum number of iterations (scalar)
% - l : reference (true) dominant eigenvalue (scalar)
%
% Output:
% - lambda: the dominant eigenvalue (scalar)
% - v: the dominant eigenvector of size n
% - it: the iteration-number at termination (scalar)
% - erreval: a vector of size it containing the history of relative eigenvalue approximation errors
% - errres: a vector of size it containing the history of residuals
function [lambda, v, it, erreval, errres] = invit(n, A, x0, sigma, eps, maxit, l)
	validate_eigenvalue_approximation_arguments(n, A, x0, sigma, eps, maxit, l);

	erreval = [];
	errres = [];

	v = x0;
	y = ones(size(x0));
	shifted_A = A - sigma * eye(n);
	[L, U, P] = lu(shifted_A);

	% Source pseudocode: http://www.netlib.org/utk/people/JackDongarra/etemplates/node96.html.
	for it = 1:1:maxit
		y = U \ (L \ (P * v));
		v = y / norm(y, Inf);

		lambda = sigma + 1 / norm(y, Inf);
		erreval = [erreval, norm(l - lambda, 1) / norm(l, 1)];
		errres = [errres, norm(A * v - lambda * v, 2)];

		if (abs(l - lambda) <= eps)
			break;
		endif		
	end

end


%%%%%%%%%%%%%%%%%%%%%%%
% Original pseudocode
% from slides:
%%%%%%%%%%%%%%%%%%%%%%%
%y = (shifted_A) \ v;
%v = y / norm(y, Inf);

%lambda = sigma + 1 / norm(y, Inf);
%erreval = [erreval, norm(l - lambda, 1) / norm(l, 1)];
%errres = [errres, norm(A * v - lambda * v, 2)];

%if (abs(l - lambda) <= eps)
%	break;
%endif