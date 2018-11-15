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