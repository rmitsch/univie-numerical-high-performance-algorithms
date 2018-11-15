# Unblocked LU decomposition.
function [A, P] = plu(A, n)
    validate_A(A);
    validate_n(n);
    
    [A, P] = pivoted_lu(A, n);
end