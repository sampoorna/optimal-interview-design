function [result, time] = greedySelect(algo, candidate_items, w1_M1, num_items, cov, test)
	%disp(algo);
	time = [];
	if numel(candidate_items) == num_items
		fprintf('same number');
		result = candidate_items;
	else
		if algo == 1
			result = acc_backward_greedy2(candidate_items, w1_M1, num_items, cov, test);
		elseif algo == 7
			result = backward_greedy(candidate_items, w1_M1, num_items, test);
		elseif algo == 3
			[result, time] = acc_fwd_greedy(candidate_items, w1_M1, num_items, test);
		elseif algo == 2
			result = acc_backward_greedy(candidate_items, w1_M1, num_items, test);
		elseif algo == 6
			result = backward_greedy2(candidate_items, w1_M1, num_items, cov, test);
		elseif algo == 8
			[result, time] = fwd_greedy(candidate_items, w1_M1, num_items, test);
		elseif algo == 4
			[result, time] = acc_fwd_greedy2(candidate_items, w1_M1, num_items, cov, test);
		elseif algo == 11
			[result, time] = fwd_greedy2(candidate_items, w1_M1, num_items, cov, test);
		end
	end
end

function result = backward_greedy(candidate_items, w1_M1, num_items, test)
	result = candidate_items; % Pool of items to discard from
	Vb = w1_M1(candidate_items, :); % Corresponding latent vectors
	alpha = zeros(numel(candidate_items), 1); % Value of the function - trace(inverse(VBVB^T))
	for i=1:numel(candidate_items)-num_items % No. of runs = no. of items to be discarded
		val = f(Vb); % Value of f() with all items still to be considered
		for j=1:numel(result)
			alpha(j) = trace(sherman_morrison(val, Vb(j, :), '-')); % f() when item j is removed
			if test == 1
				fprintf('For item %d .... %f\n', result(j), alpha(j)-trace(val))
			end
		end
		[~, ind] = min(alpha); % Min value corresponding to best item to remove
		minTrace = alpha(ind);
		if test == 1
			fprintf('best item to remove %d...%f\n', result(ind),minTrace - trace(val));
			disp(trace(val))
		end
		% Removing item from consideration
		result(ind) = [];
		Vb(ind, :) = [];
		alpha(ind) = [];
	end
end

function result = acc_backward_greedy(candidate_items, w1_M1, num_items, test)
	fresh = ones(numel(candidate_items), 1);
	alpha = zeros(numel(candidate_items), 1);
	result = candidate_items; % Pool of items to discard from
	Vb = w1_M1(candidate_items, :); % Corresponding latent vectors
	f_M = f(Vb);
	for i=1:numel(candidate_items)
		alpha(i) = g(f_M, w1_M1(candidate_items(i), :)); % Marginal gain of removing the ith candidate item from the whole matrix
	end
	
	while numel(result) > num_items
		[~, ind] = min(alpha); % Get the index of the item with the lowest marginal gain
		if test == 1
			fprintf('Considering ... %d', result(ind))
		end
		if fresh(ind) == 1 % If it is fresh, it must be the true minimum
			if test == 1
				fprintf('..Fresh!\n Removing %d.........%f size of Vb: %f\n', result(ind), alpha(ind), size(Vb, 1));
				disp(alpha');
				disp(result);
				disp(trace(f_M))
			end
			% Remove the item from consideration
			alpha(ind) = [];
			fresh(ind) = [];
			result(ind) = [];
			Vb(ind, :) = [];
			f_M = f(Vb);
			% We move to the next stage, and the alpha values are no longer fresh
			fresh = zeros(numel(fresh), 1);
		else % If it is not fresh, make it fresh
			if test == 1
				fprintf(' ...but not fresh\n', result(ind));
				disp(result);
				disp(alpha')
			end
			fresh(ind) = 1;
			alpha(ind) = g(f_M, Vb(ind, :)); % Update alpha
		end
	end
end

function result = backward_greedy2(candidate_items, w1_M1, num_items, covariance, test)
	result = candidate_items; % Pool of items to discard from
	Vb = w1_M1(candidate_items, :); % Corresponding latent vectors
	cov = covariance;
	alpha = zeros(numel(candidate_items), 1); % Value of the function - trace(inverse(VBVB^T))
	for i=1:numel(candidate_items)-num_items % No. of runs = no. of items to be discarded
		val = bgs2(Vb, cov); % Value of f() with all items still to be considered
		for j=1:numel(result)
			alpha(j) = trace(sherman_morrison(val, Vb(j, :)/sqrt(cov(j)), '-')); % f() when item j is removed
			if test == 1
				fprintf('For item %d .... %f\n', result(j), alpha(j)-trace(val))
			end
		end
		[~, ind] = min(alpha); % Min value corresponding to best item to remove
		minTrace = alpha(ind);
		if test == 1
			fprintf('best item to remove %d...%f\n', result(ind),minTrace - trace(val));
			disp(trace(val))
		end
		% Removing item from consideration
		result(ind) = [];
		Vb(ind, :) = [];
		alpha(ind) = [];
		cov(ind) = [];
	end
end

function result = acc_backward_greedy2(candidate_items, w1_M1, num_items, covariance, test)
	fresh = ones(numel(candidate_items), 1);
	alpha = zeros(numel(candidate_items), 1);
	result = candidate_items; % Pool of items to discard from
	Vb = w1_M1(candidate_items, :); % Corresponding latent vectors
	cov = covariance;
	f_M = bgs2(Vb, cov);
	for i=1:numel(candidate_items)
		alpha(i) = g(f_M, w1_M1(candidate_items(i), :)/sqrt(cov(i))); % Marginal gain of removing the ith candidate item from the whole matrix
	end
	
	while numel(result) > num_items
		[~, ind] = min(alpha); % Get the index of the item with the lowest marginal gain
		if test == 1
			fprintf('Considering ... %d', result(ind))
		end
		if fresh(ind) == 1 % If it is fresh, it must be the true minimum
			if test == 1
				fprintf('..Fresh!\n Removing %d.........%f size of Vb: %f\n', result(ind), alpha(ind), size(Vb, 1));
				disp(alpha');
				disp(result');
				disp(trace(f_M))
			end
			% Remove the item from consideration
			alpha(ind) = [];
			fresh(ind) = [];
			result(ind) = [];
			Vb(ind, :) = [];
			cov(ind) = [];
			f_M = bgs2(Vb, cov);
			% We move to the next stage, and the alpha values are no longer fresh
			fresh = zeros(numel(fresh), 1);
		else % If it is not fresh, make it fresh
			if test == 1
				fprintf(' ...but not fresh\n', result(ind));
				disp(result');
				disp(alpha')
			end
			fresh(ind) = 1;
			alpha(ind) = g(f_M, Vb(ind, :)/sqrt(cov(ind))); % Update alpha
		end
	end
end

function [result, time] = fwd_greedy(candidate_items, w1_M1, num_items, test)
	tic
	result = [];
	Vb = [];
	alpha = zeros(numel(candidate_items), 1);
	temp_items = candidate_items;
	k = size(w1_M1(1,:), 2);
	for i=1:numel(candidate_items)
		alpha(i) = trace(f(w1_M1(candidate_items(i), :))); % Value of having only each item
	end
	timeElapsed = toc;
	time = repmat(timeElapsed, [num_items, 1]);
	
	tic
	%%%%%%%%%%%%%%%%% NEW ADDITIONS %%%%%%%%%%%%%%%%%%%%%%%
	[~, ind] = min(alpha); % Alpha currently has f() with only one item, representing error, so choose the item that leads to least error
	alpha(ind) = [];
	% Add to result set
	result = [temp_items(ind)];
	Vb = [w1_M1(temp_items(ind), :)];
	timeElapsed = toc;
	time(numel(result):num_items) = time(numel(result):num_items) + timeElapsed;
	if numel(result) == num_items
		return
	end
	tic
	temp_items(ind) = [];
	f_M = f(Vb);
	% We move to the next stage, and the alpha values are no longer fresh
	for i=1:numel(temp_items)
		alpha(i) = l(f_M, w1_M1(temp_items(i), :)); % Additional value of having 2nd item
	end
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	while numel(result) < num_items
		% Get the index of the item with the max reduction in error
		[~, ind] = max(alpha); 
		% Add to result set
		result = [result, temp_items(ind)];
		Vb = [Vb; w1_M1(temp_items(ind), :)];
		timeElapsed = toc;
		time(numel(result):num_items) = time(numel(result):num_items) + timeElapsed;
		tic
		f_M = f(Vb);
		
		% Remove the item from consideration
		if test == 1
			fprintf('%f ...... adding %d\n', alpha(ind), temp_items(ind));
		end
		alpha(ind) = [];
		temp_items(ind) = [];
		
		% Update error values for each item
		for item=1:numel(temp_items)
			alpha(item) = l(f_M, w1_M1(temp_items(item), :)); % Additional value of each item
		end
	end
	if test == 1
		result
	end
end

function [result, time] = acc_fwd_greedy(candidate_items, w1_M1, num_items, test)
	tic
	result = [];
	Vb = [];
	fresh = ones(numel(candidate_items), 1);
	alpha = zeros(numel(candidate_items), 1);
	temp_items = candidate_items;
	k = size(w1_M1(1,:), 2);
	for i=1:numel(candidate_items)
		alpha(i) = trace(f(w1_M1(candidate_items(i), :))); % Value of having only each item
	end
	timeElapsed = toc;
	time = repmat(timeElapsed, [num_items, 1]);
	
	tic
	%%%%%%%%%%%%%%%%% NEW ADDITIONS %%%%%%%%%%%%%%%%%%%%%%%
	[~, ind] = min(alpha); % Alpha currently has f() with only one item, representing error, so choose the item that leads to least error
	alpha(ind) = [];
	fresh(ind) = [];
	% Add to result set
	result = [temp_items(ind)];
	Vb = [w1_M1(temp_items(ind), :)];
	timeElapsed = toc;
	time(numel(result):num_items) = time(numel(result):num_items) + timeElapsed;
	if numel(result) == num_items
		return
	end
	tic
	temp_items(ind) = [];
	f_M = f(Vb);
	% We move to the next stage, and the alpha values are no longer fresh
	fresh = zeros(numel(fresh), 1);
	for i=1:numel(temp_items)
		alpha(i) = l(f_M, w1_M1(temp_items(i), :)); % Additional value of having 2nd item
	end
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	
	while numel(result) < num_items
		[~, ind] = max(alpha); % Get the index of the item with the max reduction in error
		if fresh(ind) == 1 % If it is fresh, it must be the true maximum
			% Remove the item from consideration
			if test == 1
				fprintf('%f ...... adding %d\n', alpha(ind), temp_items(ind));
			end
			alpha(ind) = [];
			fresh(ind) = [];
			% Add to result set
			result = [result, temp_items(ind)];
			Vb = [Vb; w1_M1(temp_items(ind), :)];
			timeElapsed = toc;
			time(numel(result):num_items) = time(numel(result):num_items) + timeElapsed;
			tic
			temp_items(ind) = [];
			f_M = f(Vb);
			% We move to the next stage, and the alpha values are no longer fresh
			fresh = zeros(numel(fresh), 1);
		else % If it is not fresh, make it fresh
			fresh(ind) = 1;
			alpha(ind) = l(f_M, w1_M1(temp_items(ind), :)); % Update alpha
		end
	end
	if test == 1
		result
	end
end

function [result, time] = fwd_greedy2(candidate_items, w1_M1, num_items, covariance, test)
	tic
	alpha = zeros(numel(candidate_items), 1);
	result = [];
	Vb = [];
	cov = [];
	temp_items = candidate_items;
	k = size(w1_M1(1,:), 2);
	for i=1:numel(candidate_items)
		alpha(i) = trace(bgs2(w1_M1(candidate_items(i), :), covariance(i))); % Value of having only each item
	end
	timeElapsed = toc;
	time = repmat(timeElapsed, [num_items, 1]);
	
	tic
	%%%%%%%%%%%%%%%%% NEW ADDITIONS %%%%%%%%%%%%%%%%%%%%%%%
	[~, ind] = min(alpha); % Alpha currently has f() with only one item, representing error, so choose the item that leads to least error
	alpha(ind) = [];
	% Add to result set
	result = [temp_items(ind)];
	Vb = [w1_M1(temp_items(ind), :)];
	timeElapsed = toc;
	time(numel(result):num_items) = time(numel(result):num_items) + timeElapsed;
	if numel(result) == num_items
		return
	end
	tic
	temp_items(ind) = [];
	covariance(ind) = [];
	f_M = f(Vb);
	% We move to the next stage, and the alpha values are no longer fresh
	for i=1:numel(temp_items)
		alpha(i) = l(f_M, w1_M1(temp_items(i), :)/sqrt(covariance(i))); % Additional value of having 2nd item
	end
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	while numel(result) < num_items
		[~, ind] = max(alpha); % Get the index of the item with the max error reductino
		if test == 1
			fprintf('Considering ... %d', temp_items(ind))
		end
		
		if test == 1
			fprintf('%f ...... adding %d\n', alpha(ind), temp_items(ind));
		end
		alpha(ind) = [];
		% Add to result set
		result = [result, temp_items(ind)];
		Vb = [Vb; w1_M1(temp_items(ind), :)];
		timeElapsed = toc;
		time(numel(result):num_items) = time(numel(result):num_items) + timeElapsed;
		tic
		temp_items(ind) = [];
		covariance(ind) = [];
		f_M = f(Vb);
		
		for i=1:numel(temp_items)
			alpha(i) = l(f_M, w1_M1(temp_items(i), :)/sqrt(covariance(i))); % Additional value of having 2nd item
		end
	end
end

function [result, time] = acc_fwd_greedy2(candidate_items, w1_M1, num_items, covariance, test)
	tic
	fresh = ones(numel(candidate_items), 1);
	alpha = zeros(numel(candidate_items), 1);
	result = [];
	Vb = [];
	temp_items = candidate_items;
	k = size(w1_M1(1,:), 2);
	for i=1:numel(candidate_items)
		alpha(i) = trace(bgs2(w1_M1(candidate_items(i), :), covariance(i))); % Value of having only each item
	end
	timeElapsed = toc;
	time = repmat(timeElapsed, [num_items, 1]);
	
	tic
	%%%%%%%%%%%%%%%%% NEW ADDITIONS %%%%%%%%%%%%%%%%%%%%%%%
	[~, ind] = min(alpha); % Alpha currently has f() with only one item, representing error, so choose the item that leads to least error
	alpha(ind) = [];
	fresh(ind) = [];
	% Add to result set
	result = [temp_items(ind)];
	Vb = [w1_M1(temp_items(ind), :)];
	timeElapsed = toc;
	time(numel(result):num_items) = time(numel(result):num_items) + timeElapsed;
	if numel(result) == num_items
		return
	end
	tic
	temp_items(ind) = [];
	f_M = f(Vb);
	covariance(ind) = [];
	
	% We move to the next stage, and the alpha values are no longer fresh
	fresh = zeros(numel(fresh), 1);
	for i=1:numel(temp_items)
		alpha(i) = l(f_M, w1_M1(temp_items(i), :)/sqrt(covariance(i))); % Additional value of having 2nd item
	end
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	
	while numel(result) < num_items
		[~, ind] = max(alpha); % Get the index of the item with the max error reductino
		if test == 1
			fprintf('Considering ... %d', temp_items(ind))
		end
		if fresh(ind) == 1 % If it is fresh, it must be the true minimum
			if test == 1
				fprintf('..Fresh!\n Adding %d.........%f size of Vb: %f\n', temp_items(ind), alpha(ind), size(Vb, 1));
				disp(alpha');
				disp(result);
				disp(trace(f_M))
			end
			% Remove the item from consideration
			if test == 1
				fprintf('%f ...... adding %d\n', alpha(ind), temp_items(ind));
			end
			alpha(ind) = [];
			fresh(ind) = [];
			% Add to result set
			result = [result, temp_items(ind)];
			Vb = [Vb; w1_M1(temp_items(ind), :)];
			timeElapsed = toc;
			time(numel(result):num_items) = time(numel(result):num_items) + timeElapsed;
			tic
			temp_items(ind) = [];
			f_M = f(Vb);
			covariance(ind) = [];
			
			% We move to the next stage, and the alpha values are no longer fresh
			fresh = zeros(numel(fresh), 1);
		else % If it is not fresh, make it fresh
			if test == 1
				fprintf(' ...but not fresh\n', temp_items(ind));
				disp(result);
				disp(alpha')
			end
			fresh(ind) = 1;
			alpha(ind) = l(f_M, w1_M1(temp_items(ind), :)/sqrt(covariance(ind))); % Update alpha
		end
	end
end

function result = f(M)
	k = size(M(1,:), 2);

	%%%%% CHOICE OF LAMBDA MAKES A DIFFERENCE
	lambda = 1;
	result = pinv(M'*M + lambda*eye(k));
end

function result = bgs2(M, cov)
	k = size(M(1,:), 2);

	lambda = 1;
	result = pinv(M'*inv(diag(cov))*M + lambda*eye(k));
end

function result = g(f_M, v) % Marginal gain in error when removing an item
	result = trace(sherman_morrison(f_M, v, '-')) - trace(f_M);
end

function result = l(f_M, v) % Reduction in error when adding an item
	result = trace(f_M) - trace(sherman_morrison(f_M, v, '+'));
end

function result = sherman_morrison(A_inv, vj, c) % Rank one update through SM
	if c == '-' % Rank one subtraction
		result = A_inv + (A_inv*vj'*vj*A_inv)/(1 - vj*A_inv*vj');
	else % Rank one addition
		result = A_inv - (A_inv*vj'*vj*A_inv)/(1 + vj*A_inv*vj');
	end
end