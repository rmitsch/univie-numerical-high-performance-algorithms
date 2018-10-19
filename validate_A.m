function validate_A(A)
    if (!ismatrix(A) || !isnumeric(A))
        error ('A must be a numeric matrix.');
    endif
end