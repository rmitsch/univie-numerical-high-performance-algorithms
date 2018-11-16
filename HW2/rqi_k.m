# Rayleigh quotient iteration with shifts only in every k-th iteration.
#
# Input:
# - n: dimension (scalar)
# - A: n × n matrix
# - x0: starting vector of size n
# - sigma: shift/eigenvalue approximation (scalar)
# - eps: error tolerance (scalar)
# - maxit: the maximum number of iterations (scalar)
# - l : reference (true) dominant eigenvalue (scalar)
# - k: a scalar defining the number of k − 1 iterations before the shift is updated
#
# Output:
# - lambda: the dominant eigenvalue (scalar)
# - v: the dominant eigenvector of size n
# - it: the iteration-number at termination (scalar)
# - erreval: a vector of size it containing the history of relative eigenvalue approximation errors
# - errres: a vector of size it containing the history of residuals
function [lambda, v, it, erreval, errres] = rqi_k(n, A, x0, sigma, eps, maxit, l, k)
	lambda = 0;
	v = 0;
	it = 0;
	erreval = 0;
	errres = 0;
end