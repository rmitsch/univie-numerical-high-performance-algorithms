function validate_n(n)
    if (n <= 0 || !isscalar(n))
        error ('n must be a numeric scalar greater than zero.');
    endif
end