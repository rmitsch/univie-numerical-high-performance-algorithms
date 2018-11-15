# Blocked LU decomposition.
function [A, P] = uplu(A, n)
    validate_A(A);
    validate_n(n);
    A_orig = A;

    [A, P] = recursive_block_lu(A, n, 10);
end