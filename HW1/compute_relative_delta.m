# Computes residual for comparing L * U vs. the original matrix A.
# Note: Also used for relative error, since calculation is identical.
# Both arguments (A_prime, A) have to be of the same shape.
function delta = compute_relative_delta(A_prime, A, n)
    validate_A(A);
    validate_A(A_prime);
    validate_n(n);
    
    delta = compute_1norm(A_prime - A, n) / compute_1norm(A, n);
end