function num_ops = calc_num_ops(method, n, it, k)
	num_ops = -1;

	if (strcmp(method, 'invit') == 0)
		num_ops = 2/3 * n^3 + it * n;
	elseif (strcmp(method, 'rqi') == 0)
		num_ops = it * (n^3 + n^2 + n);
	elseif (strcmp(method, 'rqi_k') == 0)
		num_ops = it * n + (it / k) * (n^3 + n^2); 
	endif
		
end