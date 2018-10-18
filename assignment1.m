1;

### TODOS ###
# variable, not divisible block size?
# 3. perf. optimization
#   3. 1. loop order (mult. sequence?s)
#   3. 2. blocking - block size?

# Generate random non-singular matrix.
function A = generate_random_nonsingular_matrix(n)
    A = rand(n);
end

function [A, P] = pivoted_lu(A, n, r = -1)
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
            rho = k + 1:n;

            # Compute subdiagonal entries of L, store in A's subdiagonale instead of M.
            A(rho, k) /= A(k, k);

            # If this is not a rectangular matrix: Apply elimination matrix (implicitly; by using values stored in A).
            if (r == -1)
                A(rho, rho) -= A(rho, k) .* A(k, rho);    
            elseif (k < r)
                mu = k + 1:r;
                A(rho, mu) -= A(rho, k) * A(k, mu);
            endif
        end        
    end    
end

# Unblocked LU decomposition.
function [A, P] = plu(A, n)
    validate_A(A);
    validate_n(n);
    
    [A, P] = pivoted_lu(A, n);
end

# Blocked LU decomposition.
function [A, P] = uplu(A, n)
    validate_A(A);
    validate_n(n);
    A_orig = A;

    # todo What if block size b is not a divisor of A respectively n? Check for that.    
    [A, P] = recursive_block_lu(A, n, 5);

    L = tril(A, -1);
    L(1:size(L)(1) + 1:end) = 1;
    U = triu(A, 0);
    printf("delta = %f \n", compute_relative_delta(L * U, P * A_orig, n));
end

function [A, P] = recursive_block_lu(A, n, r)
    P = 0;

    if (n <= r)
        # 3.2.1
        A_orig = A;
        [A, P] = pivoted_lu(A, n, -1);
        #L = tril(A, -1);
        #L(1:size(L)(1) + 1:end) = 1;
        #U = triu(A, 0);
        #printf("inner delta = %f \n", compute_relative_delta(L * U, P * A_orig, n));
    else
        # 3.2.8: LU-decompose A(:, 1:r).
        P = eye(n);
        
        LU = A(:, 1:r); 
        A1121_orig = A(:, 1:r);
        for k = 1:r
            # Initialize permutation matrix P_k for this step.
            P_k = eye(n);

            # Fetch all elements we want to investigate to find |a_pk| >= |a_ik| -> index of maximum on or below the diagonale.
            [p_value, p_index] = max(abs(LU(k:r, k)));
            # Add offset.
            p_index += k - 1;
            
            # Swap rows if there is a bigger value underneath the diagonale.
            if p_index != k
                LU([p_index, k], :) = LU([k, p_index], :);
                
                # Save step in temporary permutation matrix.
                P_k(p_index, k) = 1;
                P_k(k, p_index) = 1;
                P_k(k, k) = 0;
                P_k(p_index, p_index) = 0;
                
                # Update permutation matrix.
                P = P_k * P;
            end        
            
            if LU(k, k) != 0
                rho = k + 1:n;
                LU(rho, k) /= LU(k, k);
              
                if (k < r)
                    mu = k + 1:r;
                    LU(rho, mu) -= LU(rho, k) * LU(k, mu);
                endif
            end        
        end        

        A1222 = P * A(:, r + 1:n);
        A12_permutated = A1222(1:r, :);
        A21_permutated = A1222(r + 1:n, :);
        
        # Extract L11, L21 and U11.
        LU11 = LU(1:r, 1:r);
        L11 = tril(LU11, -1);
        L11(1:size(L11)(1) + 1:end) = 1;
        U11 = triu(LU11, 0);
        L21 = LU(r + 1:end, 1:r);
        sum(sum([L11; L21] * U11 - P * A1121_orig))

        # Solve for U12.
        U12 = L11 \ A12_permutated; #A(1:r, r + 1:n);
        
        # Compute A_tilde.
        A_tilde = A21_permutated - L21 * U12; # A(r + 1:n, r + 1:n) - L21 * U12;

        # Repeat for submatrix; ensemble L and U.
        [A_sub, P_sub] = recursive_block_lu(A_tilde, n - r, r);
        A = [
            # L11, [0]; L21, L22
            tril(LU11, -1), zeros(r, n - r); P_sub * L21, tril(A_sub, -1)] + ...
            # U11, U12; [0], U22
            [U11, U12; zeros(n - r, r), triu(A_sub, 0)
        ];
    endif
end

function [A, P] = recursive_block_lu_X(A, n, r)
    if (n <= r)
        # 3.2.1
        [A, P] = plu(A, n);

    else
        # 3.2.8: LU-decompose A(:, 1:r).
        [A, P] = pivoted_lu(A, n, r);

        # Extract L11, L21 and U11.
        LU11 = A(1:r, 1:r);
        L11 = tril(LU11, -1);
        L11(1:size(L11)(1) + 1:end) = 1;
        U11 = triu(LU11, 0);
        L21 = A(r + 1:end, 1:r);
        
        # Solve for U12.
        U12 = L11 \ A(1:r, r + 1:n);
        
        # Compute A_tilde.
        A_tilde = A(r + 1:n, r + 1:n) - L21 * U12;

        # Repeat for remaining submatrix; ensemble L and U.
        [A_sub, P_sub] = recursive_block_lu(A_tilde, n - r, r);
        A = [
            # L11, [0]; L21, L22
            tril(LU11, -1), zeros(r, n - r); P_sub * L21, tril(A_sub, -1)] + ...
            # U11, U12; [0], U22
            [U11, U12; zeros(n - r, r), triu(A_sub, 0)
        ];
    endif
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

function [rn, foe, fae, t] = stats(A, n, lu_routine)
    # Decompose into L, U and P.
    tic_id = tic;

    [LU, P] = lu_routine(A, n);
    #return

    # Extract L and U from returned matrix.
    L = tril(LU, -1);
    U = triu(LU, 0);
    # Set diagonale of L to 1.
    L(1:1 + size(L, 1):end) = 1;

    t = toc(tic_id);
    
    # Determine b so that x is vector of ones.
    x = ones([n 1]);
    b = A * x;
    y = L \ (P * b);
    x_calc = U \ y;
    
    # Compute relative factorization error.
    fae = compute_relative_delta_with_denominator(P * A, L * U, A, n);
    rn = compute_relative_delta(A * x_calc, b, n);
    foe = compute_relative_delta(x_calc, x, n);
end

# Evaluates results of blocked LU decomposition.
function [rn, foe, fae, t] = pluStats(A, n)
    [rn, foe, fae, t] = stats(A, n, @plu);
end

# Evaluate results of unblocked LU decomposition.
function [rn, foe, fae, t] = upluStats(A, n)
    [rn, foe, fae, t] = stats(A, n, @uplu);
end

function plot_results(rn, foe, fae, t, n, block_str)
    figure('Position',[0, 0, 800, 250])
    grid on
    hold on

    # Plot residuals. Ignore warnings for now since otherwise we'll get some of them due to some deltas being 0.
    #warning('off','all');
    semilogy(n, rn, '1; Rel. residual norm;.-');
    semilogy(n, foe, "markersize", 3, '1; Rel. forward error;o-');
    semilogy(n, fae, '3; Rel. factorization error;.-');
    legend ({
            "Rel. residual norm", 
            "Rel. forward error", 
            "Rel. factorization error"
        }, "location", "eastoutside")
    title (strcat("Error metrics for ", block_str), "fontsize", 16);

    figure('Position',[0, 0, 800, 250])
    grid on
    hold off
    semilogy(n, t, "markersize", 3, '3; Runtime;o-');
    legend ({"Runtime in seconds"}, "location", "eastoutside")
    title (strcat("Runtimes for ", block_str), "fontsize", 16);
end


#####################################
# Run LU decomposition & evaluation.
#####################################


problem_sizes = 10:10:10;
rel_residuals_unblocked = zeros(size(problem_sizes));
rel_fw_errors_unblocked = zeros(size(problem_sizes));
rel_factorization_errors_unblocked = zeros(size(problem_sizes));
runtimes_unblocked = zeros(size(problem_sizes));
rel_residuals_blocked = zeros(size(problem_sizes));
rel_fw_errors_blocked = zeros(size(problem_sizes));
rel_factorization_errors_blocked = zeros(size(problem_sizes));
runtimes_blocked = zeros(size(problem_sizes));

for i = 1:size(problem_sizes)(2)
    printf('#%i\n', i);

    # Generate random non-singular matrix of rank n.
    A = generate_random_nonsingular_matrix(problem_sizes(i));

    #{
    # Execute unblocked LU decomposition.
    [ ...
        rel_residuals_unblocked(i),  ...
        rel_fw_errors_unblocked(i),  ...
        rel_factorization_errors_unblocked(i),  ...
        runtimes_unblocked(i) ...
    ] = pluStats(A, problem_sizes(i));
    #}

    # Execute blocked LU decomposition.
    [ ...
        rel_residuals_blocked(i),  ...
        rel_fw_errors_blocked(i),  ...
        rel_factorization_errors_blocked(i),  ...
        runtimes_blocked(i) ...
    ] = upluStats(A, problem_sizes(i));
end

#{
plot_results(
    rel_residuals_unblocked, 
    rel_fw_errors_unblocked, 
    rel_factorization_errors_unblocked, 
    runtimes_unblocked, 
    problem_sizes,
    " unblocked"
)
#}
#{
plot_results(
    rel_residuals_blocked, 
    rel_fw_errors_blocked, 
    rel_factorization_errors_blocked, 
    runtimes_blocked, 
    problem_sizes,
    " blocked"
)
#}