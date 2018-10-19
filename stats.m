function [rn, foe, fae, t] = stats(A, n, lu_routine)
    # Decompose into L, U and P.
    tic_id = tic;

    # Decompose A, extract L and U from returned matrix.
    [LU, P] = lu_routine(A, n);
    L = tril(LU, -1);
    U = triu(LU, 0);
    # Set diagonale of L to 1.
    L(1:1 + size(L, 1):end) = 1;

    t = toc(tic_id);
    
    # Determine b so that x is vector of ones.
    #x = ones([n 1]);
    #b = A * x;
    #y = L \ (P * b);
    #x_calc = U \ y;
    
    # Compute relative factorization error.
    #fae = compute_relative_delta_with_denominator(P * A, L * U, A, n);
    #rn = compute_relative_delta(A * x_calc, b, n);
    #foe = compute_relative_delta(x_calc, x, n);
    fae = 0;
    rn = 0;
    foe = 0;
end