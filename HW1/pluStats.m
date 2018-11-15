# Evaluates results of blocked LU decomposition.
function [rn, foe, fae, t] = pluStats(A, n)
    [rn, foe, fae, t] = stats(A, n, @plu);
end