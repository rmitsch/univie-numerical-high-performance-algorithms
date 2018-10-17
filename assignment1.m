1;

### TODOS ###
# 1. input validation
# 2. result evaluation
# 3. perf. optimization
#   3. 1. loop order (mult. sequence?s)
#   3. 2. blocking

# Generate random non-singular matrix.
function A = generate_random_nonsingular_matrix(n)
    A = rand(n);
end

# Unblocked LU decomposition.
function [A, P] = plu(A, n)
    validate_A(A);
    validate_n(n);
    
    # Initialize permutation matrix P.
    P = eye(n);
     
    for k = 1:n - 1
        # Initialize permutation matrix P_k for this step.
        P_k = eye(n);

        # Fetch all elements we want to investigate to find |a_pk| >= |a_ik| -> index of maximum on or below the diagonale.
        [p_value, p_index] = max(abs(A(k:n, k)));
        # Add offset.
        p_index += k - 1;
        
        # Swap rows if there is a bigger value underneath the diagonale.
        if p_index != k
            # Swap rows.
            A([p_index, k], :) = A([k, p_index], :);
            
            # Save step in temporary permutation matrix.
            P_k(p_index, k) = 1;
            P_k(k, p_index) = 1;
            P_k(k, k) = 0;
            P_k(p_index, p_index) = 0;
            
            # Update permutation matrix.
            P = P_k * P;
        end        
        
        if A(k, k) != 0
            # Compute subdiagonal entries of L, store in A's subdiagonale instead of M.
            A(k + 1:n, k) = (A(k + 1:n, k) / A(k, k));
            
            # Apply elimination matrix (implicitly; by using values stored in A).
            A(k + 1:n, k + 1:n) -= A(k + 1:n, k) .* A(k, k + 1:n);    
        end        
    end
end

# Computes residual for comparing L * U vs. the original matrix A.
# Note: Also used for relative error, since calculation is identical.
# Both arguments (A_prime, A) have to be of the same shape.
function delta = compute_relative_delta(A_prime, A, n)
    validate_A(A);
    validate_A(A_prime);
    validate_n(n);
    
    delta = compute_1norm(A_prime - A, n) / compute_1norm(A, n);
end

function delta = compute_relative_delta_with_denominator(A_prime, A, denominator, n)
    validate_A(A);
    validate_A(A_prime);
    validate_A(denominator);
    validate_n(n);

    delta = compute_1norm(A_prime - A, n) / compute_1norm(denominator, n);
end

function validate_n(n)
    if (n <= 0 || !isscalar(n))
        error ('n must be a numeric scalar greater than zero.');
    endif
end

function validate_A(A)
    if (!ismatrix(A) || !isnumeric(A))
        error ('A must be a numeric matrix.');
    endif
end

# Computes norm for given matrix.
function norm1 = compute_1norm(A, n)
    validate_n(n);
    validate_A(A);

    norm1 = 0;
    if !(isvector(A))
        for k = 1:n
            column_sum = sum(abs(A(:, k)));
            if column_sum > norm1
                norm1 = column_sum;
            end
        end
    else
        norm1 = sum(abs(A));
    end
end

# Generate random non-singular matrices, extract and return LU matrices.
# Note: These are _not_ the L and U matrix generated by the LU decomposition, but rather just upper/lower triangular
# matrices from randomly generated non-singular matrix.
function [L, U] = generate_random_nonsingular_triangular_matrices(n)
    validate_n(n);

    A = rand(n);
    # Generate random matrices, extract upper and lower triangular matrices.
    # Note: Extracting lower and upper triangular matrix from a randomly generated matrix led to close-to-singular 
    # matrices - why? Hence using LU to fetch L, U, P for a randomly generated matrix A.
    [L, U, P] = lu(A);
    
    # Set diagonale of L to 1.
    #L(1:1 + size(L, 1):end) = 1;
end

# Evaluates results of blocked LU decomposition.
function [rn, foe, fae, t] = pluStats(A, n)
end

# Evaluate results of unblocked LU decomposition.
function [rn, foe, fae, t] = upluStats(A, n)
end


#####################################
# Run LU decomposition & evaluation.
#####################################


problem_sizes = 100:100:500;
rel_residuals = zeros(size(problem_sizes));
rel_fw_errors = zeros(size(problem_sizes));
rel_factorization_errors = zeros(size(problem_sizes));

for i = 1:size(problem_sizes)(2)
    printf('#%i\n', i)
    n = problem_sizes(i);

    # Generate random non-singular matrix of rank n.
    A = generate_random_nonsingular_matrix(n);
    
    # Decompose into L, U and P.
    [LU, P] = plu(A, n);

    # Extract L and U from returned matrix.
    L = tril(LU, -1);
    U = triu(LU, 0);
    # Set diagonale of L to 1.
    L(1:1 + size(L, 1):end) = 1;

    # Compute relative factorization error.
    rel_factorization_errors(i) = compute_relative_delta_with_denominator(P * A, L * U, A, n);

    # Determine b so that x is vector of ones.
    x = ones([n 1]);
    b = A * x;
    y = L \ (P * b);
    x_calc = U \ y;
    
    # Solve system.
    # calc. of rel. res. correct?
    rel_residuals(i) = compute_relative_delta(A * x_calc, b, n);
    rel_fw_errors(i) = compute_relative_delta(x_calc, x, n);
end

rel_factorization_errors
rel_fw_errors
rel_residuals
