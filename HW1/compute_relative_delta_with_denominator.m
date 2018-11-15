# See compute_relative_delta(...). Uses specified denominator for normalization.
function delta = compute_relative_delta_with_denominator(A_prime, A, denominator, n)
    validate_A(A);
    validate_A(A_prime);
    validate_A(denominator);
    validate_n(n);

    delta = compute_1norm(A_prime - A, n) / compute_1norm(denominator, n);
end