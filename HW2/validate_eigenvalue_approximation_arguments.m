function validate_eigenvalue_approximation_arguments(n, A, x0, sigma, eps, maxit, l)
	validateattributes(n, {'uint32'}, {'scalar', '>', 1});
	validateattributes(A, {'numeric'}, {'nonnan', 'real', 'size', [n, n]})
	validateattributes(x0, {'numeric'}, {'nonnan', 'size', [n, 1]});
	validateattributes(sigma, {'numeric'}, {'scalar'});
	validateattributes(eps, {'numeric'}, {'scalar', 'nonnan'});
	validateattributes(maxit, {'integer'}, {'scalar', 'nonnan', '>', 0});
	validateattributes(l, {'numeric'}, {'scalar', 'nonnan'});
end