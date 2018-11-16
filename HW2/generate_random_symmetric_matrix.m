% Generates a dense square, symmetric, positive definite matrix of rank n. Taken from 
% https://math.stackexchange.com/questions/357980/how-to-generate-random-symmetric-positive-definite-matrices-using-matlab.
function [A] = generate_random_symmetric_matrix(n)
	validateattributes(n, {'uint32'}, {'scalar', '>', 1});

	do
	   	A = rand(n, n);
	   	% Copy lower triangular matrix to upper half of A.
	   	A = tril(A) + tril(A,-1)';
		A = A + double(n) * eye(n);
	until (rcond(A) > 0.5)
end