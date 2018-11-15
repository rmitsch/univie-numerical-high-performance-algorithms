# Solves blocked LU decomposition recursively.
function [A, P] = recursive_block_lu(A, n, r)
    if (n <= r)
        # 3.2.1
        [A, P] = pivoted_lu(A, n, -1);

    else
        # 3.2.8: LU-decompose A(:, 1:r).
        [A(:, 1:r), P] = pivoted_lu(A(:, 1:r), n, r);

        # Get permutated versions of A12 and A21.
        A1222_P = P * A(:, r + 1:n);
        A12_P = A1222_P(1:r, :);
        A21_P = A1222_P(r + 1:n, :);
        
        # Extract L11, L21 and U11.
        LU11 = A(1:r, 1:r);
        L11 = tril(LU11, -1);
        L11(1:size(L11)(1) + 1:end) = 1;
        U11 = triu(LU11, 0);
        L21 = A(r + 1:end, 1:r);

        # Solve for U12.
        U12 = L11 \ A12_P;
        
        # Compute A_tilde.
        A_tilde = A21_P - L21 * U12;

        # Repeat for submatrix; ensemble L and U.
        [A_sub, P_sub] = recursive_block_lu(A_tilde, n - r, r);
        A = [
            # L11, [0]; L21, L22
            tril(LU11, -1), zeros(r, n - r); P_sub * L21, tril(A_sub, -1)] + ...
            # U11, U12; [0], U22
            [U11, U12; zeros(n - r, r), triu(A_sub, 0)
        ]; 

        # Update permutation matrix with sub-matrix' permutation matrix.
        P_prime = eye(n);
        P_prime(r + 1:n, r + 1:n) = P_sub;
        P = P_prime * P;
    endif
end