# Apply unblocked pivoted LU decomposition.
function [A, P] = pivoted_lu(A, n, r = -1)
    max_row = n - 1;
    max_row_to_investigate = n;
    if (r != -1)
        max_row = r;
        max_row_to_investigate = r;
    endif

    # Initialize permutation matrix P.
    P = eye(n);
     
    for k = 1:max_row
        # Initialize permutation matrix P_k for this step.
        P_k = eye(n);

        # Fetch all elements we want to investigate to find |a_pk| >= |a_ik| -> index of maximum on or below the diagonale.
        [p_value, p_index] = max(abs(A(k:max_row_to_investigate, k)));
        # Add offset.
        p_index += k - 1;
        
        # Swap rows if there is a bigger value underneath the diagonale.
        if (p_index != k)
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
        
        if (A(k, k) != 0)
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