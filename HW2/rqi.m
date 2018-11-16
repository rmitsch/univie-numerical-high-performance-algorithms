# Rayleigh quotient iteration.
#
# Input:
# - n: dimension (scalar)
# - A: n × n matrix
# - x0: starting vector of size n
# - sigma: shift/eigenvalue approximation (scalar)
# - eps: error tolerance (scalar)
# - maxit: the maximum number of iterations (scalar)
# - l : reference (true) dominant eigenvalue (scalar)
#
# Output:
# - lambda: the dominant eigenvalue (scalar)
# - v: the dominant eigenvector of size n
# - it: the iteration-number at termination (scalar)
# - erreval: a vector of size it containing the history of relative eigenvalue approximation errors
# - errres: a vector of size it containing the history of residuals
function [lambda, v, it, erreval, errres] = rqi(n, A, x0, sigma, eps, maxit, l)
	lambda = 0;
	v = 0;
	it = 0;
	erreval = 0;
	errres = 0;

	v = x0;
	y = x0;
	I = eye(n);

	for it = 1:1:maxit
		sigma = (transpose(v) * A * v) / (transpose(v) * v);
		y = (A - sigma * I) \ v;
		y_norm = norm(y, Inf); 
		v = y / y_norm;
		lambda = sigma + 1 / y_norm;

		erreval = [erreval, norm(l - lambda, 1) / norm(l, 1)];
		errres = [errres, norm(A * v - lambda * v, 2)];

		if (abs(l - lambda) <= eps)
			break;
		endif
	end
end