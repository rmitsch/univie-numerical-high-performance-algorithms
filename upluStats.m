# Evaluate results of unblocked LU decomposition.
function [rn, foe, fae, t] = upluStats(A, n)
    [rn, foe, fae, t] = stats(A, n, @uplu);
end