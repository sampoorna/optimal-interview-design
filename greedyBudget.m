%%%%%%%%%%% README %%%%%%%%%%%
%The main difference between greedyBudget_testSetPaper.m and greedyBudget.m is in the way the test set is selected. 
%In this script, the test set is fixed for every algorithm, whereas in the other script, the test set is taken to be all the items that are not asked of the user during the interview. 
%We believe the method of choosing a fixed test set is the fairest, whereas in the other method, the test set items are the items considered "undesirable" by the selection algorithms.
% This script also takes less time, since the test set does not need to be recomputed at each iteration for each algorithm.

%function greedyBudget(retrain, reload, dataset, cont)
%%%%%%%%%%% OPTIONS %%%%%%%%%%
retrain = 0; % retrain = 1 when covariance needs to be computed again
reload = 1; % reload = 1 when variables need to be loaded in the MATLAB workspace
dataset = 1; % dataset = 1 for netflix, 2 for jester, 3 for ml-1m, 4 for ml-100k, 5 for ml-20m, 6 for epinions
cont = 1; %cont = 1 to validate against P*Q instead of actual ratings in the database, 0 otherwise
NUM_FACTORS = 20;
gd = 1;
runTime = tic;
%%%%%%%%%%% LOADING FILES %%%%%%%%%
if reload == 1
	fprintf('Loading files ..........\n'); 
	if dataset == 1 %%% Netflix
		origdirec = '';
		MAX_ITEMS = 17770;
	elseif dataset == 2 %%% Jester
		load '../rating_matrix_all.mat';
		load '../jester_data_withoutrat_randcold.mat';
		load '../jester_w1_M1_20.mat';
	elseif dataset == 3 %%% MovieLens 1M
		origdirec = ''; 
		MAX_ITEMS = 3952;
	elseif dataset == 4 %%% MovieLens 100K
		origdirec = ''; 
		MAX_ITEMS = 1682;
	elseif dataset == 5 %%% MovieLens 20M
		origdirec = '';
		MAX_ITEMS = 131262;
	elseif dataset == 6 %%% Epinions
		origdirec = '';
		MAX_ITEMS = 139738;
	end
	load(strcat(origdirec, 'data_withoutrat_randcold2.mat'));
	load(strcat(origdirec, 'w1_M1_', num2str(NUM_FACTORS), '.mat'));
	if gd == 1
		load(strcat(origdirec, 'gd_w1_P1_', num2str(NUM_FACTORS), '.mat'));
	else
		load(strcat(origdirec, 'true_w1_P1_', num2str(NUM_FACTORS), '.mat'));
	end
	load(strcat(origdirec, 'covariance_cold_users_', num2str(NUM_FACTORS), '.mat'));
	load(strcat(origdirec, 'ent0_entpop_pop.mat'));
	MIN_RATING = min(warm(:, 3));
	MAX_RATING = max(warm(:, 3));
	fprintf('Loading files .......... COMPLETE\n'); 
end

%%%%%%%%%%% INITIALIZATION %%%%%%%%%
fprintf('Initializing ..........\n');
%lambda  = [20, 20, 20]; % (1, 5, 5) & (5, 10, 10) & (10, 10, 10) gives good random performance, (2, 2, 2) & (2, 2, 10) gives peaking behavior --- for ML 1M
lambda  = [0.5, 0.5, 1]; %
%lambda  = [2, 2, 2]; %
num_items = 100; % Ask these many questions
BUDGET = 200; % Number of cold users to interview at a time
num_iter = 3; % Number of iterations to average over for the random baseline
algos = [1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14];

mean_rating = mean(warm(:, 3));
cold_users = unique(cold(:, 2));
all_items = unique(warm(:, 1)); 
warm_users_count = numel(unique(warm(:, 2)));

result = cell(max(algos), 1);
true_ratings = cell(max(algos), 1);
P_estimate = zeros(max(algos), NUM_FACTORS);
P_error = zeros(num_items, max(algos));
RMSE = zeros(num_items, max(algos));
bg2time = zeros(num_items, 2);
fgtime = zeros(num_items, 2);
fg2time = zeros(num_items, 2);
bgtime = zeros(num_items, 2);
user_size_accuracy = zeros(BUDGET, num_items); % Matrix of number of items rated by user v/s accuracy by algo
%bgtimeAcc = zeros(num_items, 2);
training_error = zeros(num_items, max(algos));
functionval = zeros(num_items, max(algos));
test = 0;
count = 0;
avg_per_user = 0;
fidxs = [];
stepsize = 5;
cold_data = [];
w = [0.5, 1, 1, 1, 1, 1];
fprintf('Initializing .......... COMPLETE\n');
fprintf('Running cont = %d experiments on dataset %s with %d max items, %d cold users and %d algos\n', cont, origdirec, num_items, BUDGET, numel(algos));

%%% Get list of items rated by each cold user
if retrain == 1
	
	
	%%% Get covariances of user ratings
	load(strcat(origdirec, 'w1_P1_', num2str(NUM_FACTORS), '.mat'));
	cov = zeros(MAX_ITEMS, 1);
	fprintf('Calculating covariance ..........\n');
	count = 0;
	for item=1:MAX_ITEMS
		fprintf('%d\n', item)
		warm_users_indices = find(warm(:, 1) == item);
		Pu = w1_P1(warm(warm_users_indices, 2), :);
		est_rating = (Pu*w1_M1(item, :)' + mean_rating);
		est_rating(est_rating > 5) = 5;
		est_rating(est_rating < 1) = 1;
		cov(item) = sum((est_rating - warm(warm_users_indices, 3)).^2)/numel(warm_users_indices);
		if cov(item) > (1-mean_rating).^2
			fprintf('%d....%f\n', item, cov(item))
			est_rating
			warm(warm_users_indices, 3)
			count = count +1;
		end
	end
	count
	 fprintf('Calculating covariance ..........COMPLETE\n');
	 save(strcat(origdirec, 'covariance_cold_users_', num2str(NUM_FACTORS), '.mat'), 'cov');
	if gd == 1
		load(strcat(origdirec, 'gd_w1_P1_', num2str(NUM_FACTORS), '.mat'))
	else
		load(strcat(origdirec, 'true_w1_P1_', num2str(NUM_FACTORS), '.mat'))
	end
	 
end

%%% Randomly generate new test user set
if dataset == 6
	cold_users = [7, 28 , 41 , 45 , 94, 157, 169, 181, 194, 195, 216, 232, 271, 272, 287  303, 319, 324, 336, 341, 366, 389, 422, 424, 440, 443, 456, 467, 495, 517   579, 646, 684, 727, 748, 751, 769, 776, 782, 800, 843, 949, 973, 1051, 1077, 1081, 1097,  1102,  1109,  1293,  1308,  1360,  1429,  1574,  1617,  1752,  1773,  1776,  1803,  2011    2565,  2798,  2879, 3005, 3449,  5228,  7610, 10230];
	cold_subset = randsample(cold_users, BUDGET);
else
	cold_subset = randsample(cold_users, BUDGET);
end

if retrain == 1
	fprintf('Running Ent0, Ent-pop and Popular baselines...\n');
	%%% For Ent0, Ent-pop and Popular baselines
	if any(algos == 12) == 1
		if test == 1 
			fprintf('Printing candidate items ... ');
			candidate_items
		end
		entropy0_score = zeros(MAX_ITEMS, 1);
		entropy_score = zeros(MAX_ITEMS, 1);
		ent0_items = (1:MAX_ITEMS);
		ent_items = (1:MAX_ITEMS);
		popular = zeros(MAX_ITEMS, 1);
		pop_items = (1:MAX_ITEMS);
		p = [];
		for item=1:MAX_ITEMS'
			if mod(item, 100) == 0
				fprintf('Item ... %d\n', item);
			end
			item_ratings = find(warm(:, 1) == item);
			total_ratings = nnz(item_ratings);
			if total_ratings > 0
				p = histc(warm(item_ratings, 3), (1:MAX_RATING));
				p = reshape(p, [1, MAX_RATING]);
				
				p_ent0 = p/warm_users_count;
				p = p/total_ratings;
				p0 = 1 - sum(p_ent0);
				% Find all rating categories that are empty
				ind = find(p_ent0 == 0);
				% log(1) = 0, hence the corresponding summand = 0
				p_ent0(ind) = 1;
				p(ind) = 1;
				% Find entries that will give NaN results, replace them
				if p0 == 0
					p0 = 1;
				end

				ent = - (sum(w(ceil(MIN_RATING)+1:MAX_RATING+1).*p_ent0.*log(p_ent0)));
				%if ent > min(entropy0_score)
					% Replace the minimum score by the new score
				%	index = find(entropy0_score == min(entropy0_score), 1);
					entropy0_score(item) = ent - p0*w(1)*log(p0);
					%ent0_items(index) = item;
				%elseif isnan(ent) == 1 && test == 1
				%	fprintf('Found NaN ...\n');
				%	p
				%end
				
				%if ent*log(total_ratings) > min(entropy_score)
					% Replace the minimum score by the new score
				%	index = find(entropy_score == min(entropy_score), 1);
					entropy_score(item) = - sum(p.*log(p));
				%	ent_items(index) = item;
				%end
				
				%if total_ratings > min(popular)
					% fprintf('Replacing min popular by %d .... %f by .... %f...\n', i, min(popular), total_ratings);
					% Replace the minimum score by the new score
				%	index = find(popular == min(popular), 1);
					popular(item) = total_ratings;
				%	pop_items(index) = item;
				%end
			%else
			%	index = find(candidate_items == item);
			%	candidate_items(index) = [];
			end
		end
		
		[~, SortIndex] = sort(entropy0_score);
		ent0_items = ent0_items(SortIndex);
					
		[~, SortIndex] = sort(entropy_score);
		ent_items = ent_items(SortIndex);
		
		[~, SortIndex] = sort(popular);
		pop_items = pop_items(SortIndex);

	end
end

	
	%%% Select items for each user in subset
	for user=1:BUDGET
		fprintf('User %d......\n', user);
		
		fidxs = [fidxs; num_items];
		% Get labels of test set samples
		if cont == 1
			candidate_items = all_items;%randsample(all_items, 400);
			[row, col] = find(isnan(cov(candidate_items)));
			candidate_items(row) = [];
			[row, col] = find(cov(candidate_items) == 0);
			candidate_items(row) = [];
			test_items = randsample(candidate_items, round(0.5*numel(candidate_items)));
			candidate_items = setdiff(candidate_items, test_items);
			%true_P =  w1_P1(cold_subset(user), :);
			while numel(candidate_items) < num_items + 20
				ind = find(cold_users == cold_subset(user));
				cold_users(ind) = [];
				cold_subset(user) = randsample(setdiff(cold_users, cold_subset), 1);
				candidate_items = all_items;%randsample(all_items, 3*num_items);
				[row, col] = find(isnan(cov(candidate_items)));
				candidate_items(row) = [];
				test_items = randsample(candidate_items, round(0.5*numel(candidate_items)));
				candidate_items = setdiff(candidate_items, test_items);
			end
			true_P =  w1_P1(cold_subset(user), :);
			test_ratings = true_P*w1_M1(test_items, :)' + mean_rating; %
			test_ratings(test_ratings > MAX_RATING) = MAX_RATING;
			test_ratings(test_ratings < MIN_RATING) = MIN_RATING;
			
		else
			% If the chosen cold user has not rated enough items, remove from cold users pool
			candidate_items = cold(find(cold(:, 2) == cold_subset(user)), 1); %all_items{cold_subset(user)};
			% Remove items that we haven't seen in the training data
			[row, col] = find(isnan(cov(candidate_items)));
			candidate_items(row) = [];
			high_cov = find(cov(candidate_items) > 1);
			%cov(high_cov) = [];
			%high_cov_items = find(candidate_items == high_cov);
			candidate_items(high_cov) = [];
			% Randomly select training and validation samples
			test_items = randsample(candidate_items, max(round(0.5*numel(candidate_items)), numel(candidate_items)-1000));
			candidate_items = setdiff(candidate_items, test_items);
			
			cold_data = cold(find(cold(:, 2) == cold_subset(user)), :);
			while numel(candidate_items) < num_items + 20
				ind = find(cold_users == cold_subset(user));
				cold_users(ind) = [];
				cold_subset(user) = randsample(setdiff(cold_users, cold_subset), 1);
				candidate_items = cold(find(cold(:, 2) == cold_subset(user)), 1);
				[row, col] = find(isnan(cov(candidate_items)));
				candidate_items(row) = [];
				high_cov = find(cov(candidate_items) > 1);
				candidate_items(high_cov) = [];
				test_items = randsample(candidate_items, round(0.5*numel(candidate_items)));
				candidate_items = setdiff(candidate_items, test_items);
				% Remove items that we haven't seen in the training data
				
				cold_data = cold(find(cold(:, 2) == cold_subset(user)), :);
			end
			test_ratings = cold_data(find(ismember(cold_data(:, 1), test_items)), 3)';
			test_items = sort(test_items);
			%R(cold_subset(user), test_items);
		end
		
		avg_per_user = avg_per_user + numel(candidate_items);
		user_size_accuracy(user, 1) = numel(candidate_items);
		for a=algos
			%%% Accelerated backward greedy select version 2
			if a == 1 || a == 6 || a == 4 || a == 11
				fprintf('Running BG2 OR FG2...\n');
				
				tic
				[result{a}, time] = greedySelect(a, candidate_items, w1_M1, num_items, cov(candidate_items), test);
				timeElapsed = toc;
				
				if a == 1
					result{a} = result{a}';
					bg2time(:, 1) = bg2time(:, 1) + timeElapsed; %%% ABG2
				elseif a == 6
					result{a} = result{a}';
					bg2time(:, 2) = bg2time(:, 2) + timeElapsed; %%% BG2
				elseif a == 4
					fg2time(:, 1) = fg2time(:, 1) + time; %%% AFG2
				else
					fg2time(:, 2) = fg2time(:, 2) + time; %%% FG2
				end
				sorted_result = sort(result{a});
				if cont == 1
					true_ratings{a} = true_P*w1_M1(sorted_result, :)' + mean_rating;
				else
					true_ratings{a} = cold_data(find(ismember(cold_data(:, 1), result{a}')), 3)';%R(cold_subset(user), result{a}');
				end
				P_estimate(a, :) = (lambda(3) * eye(NUM_FACTORS) + w1_M1(sorted_result, :)'* inv(diag(cov(sorted_result)))* w1_M1(sorted_result, :)) \ w1_M1(sorted_result, :)' *inv(diag(cov(sorted_result)))* (true_ratings{a} - mean_rating)';

				functionval(num_items, a) = functionval(num_items, a) + trace(inv(w1_M1(sorted_result, :)'*inv(diag(cov(sorted_result)))*w1_M1(sorted_result, :) + lambda(3)*eye(NUM_FACTORS)));
				
			%%% Accelerated backward greedy select
			elseif a == 2 || a == 7
				fprintf('Running accelerated backward...\n');
				tic
				[result{a}, ~] = greedySelect(a, candidate_items, w1_M1, num_items, cov(candidate_items), test);
				timeElapsed = toc;
				if a == 2
					bgtime(:, 1) = bgtime(:, 1) + timeElapsed;
				else
					bgtime(:, 2) = bgtime(:, 2) + timeElapsed;
				end
				result{a} = sort(result{a});
				if cont == 1
					true_ratings{a} = true_P*w1_M1(result{a}, :)' + mean_rating;
				else
					true_ratings{a} = cold_data(find(ismember(cold_data(:, 1), result{a}')), 3)';
				end
				
				P_estimate(a, :) = (lambda(2) * eye(NUM_FACTORS) + w1_M1(result{a}, :)' * w1_M1(result{a}, :)) \ w1_M1(result{a}, :)' * (true_ratings{a} - mean_rating)';
				
				functionval(num_items, a) = functionval(num_items, a) + trace(inv(w1_M1(result{a}, :)'*w1_M1(result{a}, :) + lambda(2)*eye(NUM_FACTORS)));
			%%% Accelerated Forward
			elseif a == 3 || a == 8
				fprintf('Running accelerated forward...\n');
				[result{a}, x] = greedySelect(a, candidate_items, w1_M1, num_items, cov(candidate_items), test);
				
                % We only need to time it once. Returns: A matrix.
				if a == 3
					fgtime(:, 1) = fgtime(:, 1) + x; %%% AFG
				elseif a == 8
					fgtime(:, 2) = fgtime(:, 2) + x; %%% FG
				end
				sorted_result = sort(result{a});
				if cont == 1
					true_ratings{a} = true_P*w1_M1(sorted_result, :)' + mean_rating;
				else
					true_ratings{a} = cold_data(find(ismember(cold_data(:, 1), result{a}')), 3)';%R(cold_subset(user), result{a}');
				end
				
				P_estimate(a, :) = (lambda(2) * eye(NUM_FACTORS) + w1_M1(sorted_result, :)' * w1_M1(sorted_result, :)) \ w1_M1(sorted_result, :)' * (true_ratings{a} - mean_rating)';
				
				functionval(num_items, a) = functionval(num_items, a) + trace(inv(w1_M1(sorted_result, :)'*w1_M1(sorted_result, :) + lambda(2)*eye(NUM_FACTORS)));
			%%% Random
			elseif a == 10
				fprintf('Running random...\n');
				for iter=1:num_iter
				rng('shuffle');
				result{a} = randsample(candidate_items, num_items);

				result{a} = sort(result{a});
				if cont == 1
					true_ratings{a} = true_P*w1_M1(result{a}, :)' + mean_rating;
				else
					true_ratings{a} = cold_data(find(ismember(cold_data(:, 1), result{a}')), 3)';%R(cold_subset(user), result{a}');
				end
				P_estimate(a, :) = (lambda(1) * eye(NUM_FACTORS) + w1_M1(result{a}, :)' * w1_M1(result{a}, :)) \ w1_M1(result{a}, :)' * (true_ratings{a} - mean_rating)';
				
				%functionval(num_items, a) = functionval(num_items, a) + trace(inv(w1_M1(result{a}, :)'*w1_M1(result{a}, :) + lambda(1)*eye(NUM_FACTORS)));
				
				est_rating = P_estimate(a, :)*w1_M1(test_items, :)' + mean_rating;
				est_rating(est_rating > MAX_RATING) = MAX_RATING;
				est_rating(est_rating < MIN_RATING) = MIN_RATING;
				n = sqrt(sum((est_rating - test_ratings).^2)/numel(test_items));
				RMSE(numel(result{a}), a) = RMSE(numel(result{a}), a) + n;
				P_error(numel(result{a}), a) = P_error(numel(result{a}), a) + norm(P_estimate(a, :) - w1_P1(cold_subset(user), :));
				end
			%%% High variance items
			elseif a == 5
				[~, sortingIndices] = sort(cov(candidate_items),'descend');
				topPicks = sortingIndices(2:num_items+1);
				result{a} = candidate_items(topPicks);
				sorted_result = sort(result{a});
				if cont == 1
					true_ratings{a} = true_P*w1_M1(sorted_result, :)' + mean_rating;
				else
					true_ratings{a} = cold_data(find(ismember(cold_data(:, 1), sorted_result')), 3)';%R(cold_subset(user), result{a}');
				end
				P_estimate(a, :) = (lambda(1) * eye(NUM_FACTORS) + w1_M1(sorted_result, :)' * w1_M1(sorted_result, :)) \ w1_M1(sorted_result, :)' * (true_ratings{a} - mean_rating)';
				
				functionval(num_items, a) = functionval(num_items, a) + trace(inv(w1_M1(sorted_result, :)'*w1_M1(sorted_result, :) + lambda(1)*eye(NUM_FACTORS)));
			elseif a == 9
				fprintf('Running opposite of backward greedy...\n');
				tic
				[result{a}, ~] = greedySelect(1, candidate_items, w1_M1, numel(candidate_items) - num_items, cov(candidate_items), test);
				timeElapsed = toc;

				result{a} = setdiff(candidate_items, result{a});
				
				if cont == 1
					true_ratings{a} = true_P*w1_M1(result{a}, :)' + mean_rating;
				else
					true_ratings{a} = cold_data(find(ismember(cold_data(:, 1), result{a}')), 3)';
				end
				
				P_estimate(a, :) = (lambda(3) * eye(NUM_FACTORS) + w1_M1(result{a}, :)'* inv(diag(cov(result{a})))* w1_M1(result{a}, :)) \ w1_M1(result{a}, :)' *inv(diag(cov(result{a})))* (true_ratings{a} - mean_rating)';

				functionval(num_items, a) = functionval(num_items, a) + trace(inv(w1_M1(result{a}, :)'*inv(diag(cov(result{a})))*w1_M1(result{a}, :) + lambda(3)*eye(NUM_FACTORS)));
			elseif a == 12
				fprintf('Running Entropy0 and EntPop ...\n');
				[~, ia, ib] = intersect(ent0_items, candidate_items);
				temp_candidate_items = ent0_items(sort(ia, 'descend'));
				result{a} = temp_candidate_items(1:num_items);
				
				[~, ia, ib] = intersect(ent_items, candidate_items);
				temp_candidate_items = ent_items(sort(ia, 'descend'));
				result{a+1} = temp_candidate_items(1:num_items);
				% if test == 1 
					% fprintf('Printing candidate items ... ');
					% candidate_items
				% end
				% entropy0_score = zeros(num_items, 1);
				% entropy_score = zeros(num_items, 1);
				% result{a} = zeros(num_items, 1);
				% result{a+1} = zeros(num_items, 1);
				% p = [];
				% temp_candidate_items = candidate_items;
				% for item=temp_candidate_items'
					% total_ratings = nnz(find(warm(:, 1) == item));
					% if total_ratings > 0
						% p(1) = nnz(find(warm(find(warm(:, 1) == item), 3) == 1))/warm_users_count;
						% p(2) = nnz(find(warm(find(warm(:, 1) == item), 3) == 2))/warm_users_count;
						% p(3) = nnz(find(warm(find(warm(:, 1) == item), 3) == 3))/warm_users_count;
						% p(4) = nnz(find(warm(find(warm(:, 1) == item), 3) == 4))/warm_users_count;
						% p(5) = nnz(find(warm(find(warm(:, 1) == item), 3) == 5))/warm_users_count;
						% p0 = 1 - sum(p);
						% % Find entries that will give NaN results, replace them
						% if p0 == 0
							% p0 = 1;
						% end
						% index = find(p == 0);
						% % log(1) = 0, hence the corresponding summand = 0
						% p(index) = 1;
						% ent = - (p0*w(1)*log(p0) + p(1)*w(2)*log(p(1)) + p(2)*w(3)*log(p(2)) + p(3)*w(4)*log(p(3)) + p(4)*w(5)*log(p(4)) + p(5)*w(6)*log(p(5)));
						% if ent > min(entropy0_score)
							% % Replace the minimum score by the new score
							% index = find(entropy0_score == min(entropy0_score), 1);
							% entropy0_score(index) = ent;
							% result{a}(index) = item;
						% elseif isnan(ent) == 1 && test == 1
							% fprintf('Found NaN ...\n');
							% p
						% end
						
						% if ent*log(total_ratings) > min(entropy_score)
							% % Replace the minimum score by the new score
							% index = find(entropy_score == min(entropy_score), 1);
							% entropy_score(index) = ent*log(total_ratings);
							% result{a+1}(index) = item;
						% end
					% else
						% index = find(candidate_items == item);
						% candidate_items(index) = [];
					% end
				% end
				
				%[~, SortIndex] = sort(entropy0_score);
				%result{a} = result{a}(SortIndex);
				
				%[~, SortIndex] = sort(entropy_score);
				%result{a+1} = result{a+1}(SortIndex);

				sorted_result = sort(result{a});
				sorted_result2 = sort(result{a+1});
				if cont == 1
					true_ratings{a} = true_P*w1_M1(sorted_result, :)' + mean_rating;
					true_ratings{a+1} = true_P*w1_M1(sorted_result2, :)' + mean_rating;
				else
					true_ratings{a} = cold_data(find(ismember(cold_data(:, 1), sorted_result')), 3)';
					true_ratings{a+1} = cold_data(find(ismember(cold_data(:, 1), sorted_result2')), 3)';
				end
				
				P_estimate(a, :) = (lambda(1) * eye(NUM_FACTORS) + w1_M1(sorted_result, :)'* w1_M1(sorted_result, :)) \ w1_M1(sorted_result, :)'* (true_ratings{a} - mean_rating)';
				P_estimate(a+1, :) = (lambda(1) * eye(NUM_FACTORS) + w1_M1(sorted_result2, :)'*w1_M1(sorted_result2, :)) \ w1_M1(sorted_result2, :)'* (true_ratings{a+1} - mean_rating)';
			elseif a == 14
				fprintf('Running popular...\n');
				
				[~, ia, ib] = intersect(pop_items, candidate_items);
				temp_candidate_items = pop_items(sort(ia, 'descend'));
				result{a} = temp_candidate_items(1:num_items);
				
				% if test == 1 
					% fprintf('Printing candidate items ... ');
					% candidate_items
				% end
				% popular = zeros(num_items, 1);
				% result{a} = zeros(num_items, 1);
				% for item=candidate_items'
					% total_ratings = nnz(find(warm(:, 1) == item));
					% if total_ratings > min(popular)
						% % fprintf('Replacing min popular by %d .... %f by .... %f...\n', i, min(popular), total_ratings);
						% % Replace the minimum score by the new score
						% index = find(popular == min(popular), 1);
						% popular(index) = total_ratings;
						% result{a}(index) = item;
					% end
				% end
				
				% [~, SortIndex] = sort(popular);
				% result{a} = result{a}(SortIndex);
				
				sorted_result = sort(result{a});
				
				if cont == 1
					true_ratings{a} = true_P*w1_M1(sorted_result, :)' + mean_rating;
				else
					true_ratings{a} = cold_data(find(ismember(cold_data(:, 1), sorted_result')), 3)';
				end
				
				P_estimate(a, :) = (lambda(1) * eye(NUM_FACTORS) + w1_M1(sorted_result, :)'* w1_M1(sorted_result, :)) \ w1_M1(sorted_result, :)'* (true_ratings{a} - mean_rating)';
			end
		end
		
		%%% Compute error
		for i=1:stepsize:num_items-1
			fprintf('%f..........%d\n', size(result{1}, 2), num_items-i);
			fidxs = [fidxs; num_items-i];

			for a=algos
				est_rating = P_estimate(a, :)*w1_M1(test_items, :)' + mean_rating;

                est_rating(est_rating > MAX_RATING) = MAX_RATING;
                est_rating(est_rating < MIN_RATING) = MIN_RATING;
				if a ~= 10
					n = sqrt(sum((est_rating - test_ratings).^2)/numel(test_items));
					user_size_accuracy(user, numel(result{a})) = n;
					if n > 3
						est_rating
						test_ratings
						return
					end
					RMSE(numel(result{a}), a) = RMSE(numel(result{a}), a) + n;
					P_error(numel(result{a}), a) = P_error(numel(result{a}), a) + norm(P_estimate(a, :) - w1_P1(cold_subset(user), :));
					training_error(numel(result{a}), a) = training_error(numel(result{a}), a) + sqrt(sum(((P_estimate(a, :)*w1_M1(result{a}, :)' + mean_rating) - true_ratings{a}).^2)/numel(result{a}));
				end

				if a == 1 || a == 6
					temp_candidate_items = result{a}';
					tic
					[result{a}, ~] = greedySelect(a, temp_candidate_items, w1_M1, num_items-i, cov(temp_candidate_items), test);
					timeElapsed = toc;
					result{a} = result{a}';
					if a == 1
						bg2time(1:num_items-i, 1) = bg2time(1:num_items-i, 1) + timeElapsed;
					else
						bg2time(1:num_items-i, 2) = bg2time(1:num_items-i, 2) + timeElapsed;
					end
					result{a} = sort(result{a});
					if cont == 1
						true_ratings{a} = true_P*w1_M1(result{a}, :)' + mean_rating;
					else
						true_ratings{a} = cold_data(find(ismember(cold_data(:, 1), result{a}')), 3)';%R(cold_subset(user), result{a}');
					end
					
					P_estimate(a, :) = (lambda(3)* eye(NUM_FACTORS) + w1_M1(result{a}, :)'* inv(diag(cov(result{a})))* w1_M1(result{a}, :)) \ w1_M1(result{a}, :)' *inv(diag(cov(result{a})))*(true_ratings{a} - mean_rating)';
					
					functionval(numel(result{a}), a) = functionval(numel(result{a}), a) + trace(inv(w1_M1(result{a}, :)'*inv(diag(cov(result{a})))*w1_M1(result{a}, :) + lambda(3)*eye(NUM_FACTORS)));
					P_estimate(a, :);
					
				elseif a == 2 || a == 7
					temp_candidate_items = result{a};
					tic
					[result{a}, ~] = greedySelect(a, temp_candidate_items, w1_M1, num_items-i, cov(temp_candidate_items), test);
					timeElapsed = toc;
					if a == 2
						bgtime(1:num_items-i, 1) = bgtime(1:num_items-i, 1) + timeElapsed;
					else
						bgtime(1:num_items-i, 2) = bgtime(1:num_items-i, 2) + timeElapsed;
					end
					result{a} = sort(result{a});
					if cont == 1
						true_ratings{a} = true_P*w1_M1(result{a}, :)' + mean_rating;
					else
						true_ratings{a} = cold_data(find(ismember(cold_data(:, 1), result{a}')), 3)';%R(cold_subset(user), result{a}');
					end
					
					P_estimate(a, :) = (lambda(2) * eye(NUM_FACTORS) + w1_M1(result{a}, :)' * w1_M1(result{a}, :)) \ w1_M1(result{a}, :)' * (true_ratings{a} - mean_rating)';
					
					functionval(numel(result{a}), a) = functionval(numel(result{a}), a) + trace(inv(w1_M1(result{a}, :)'*w1_M1(result{a}, :) + lambda(2)*eye(NUM_FACTORS)));
				elseif a == 3 || a == 5 || a == 8 || a == 4 || a == 11 || a == 12 || a == 13 || a == 14
					if a == 12
						result{a}(1:num_items-i);
					end

					result{a} = result{a}(1:num_items-i);
					sorted_result = sort(result{a});

					if cont == 1
						true_ratings{a} = true_P*w1_M1(sorted_result, :)' + mean_rating;
					else
						true_ratings{a} = cold_data(find(ismember(cold_data(:, 1), sorted_result')), 3)';%R(cold_subset(user), result{a}');
					end

					if a == 4 || a == 11
						P_estimate(a, :) = (lambda(3)* eye(NUM_FACTORS) + w1_M1(sorted_result, :)'* inv(diag(cov(sorted_result)))* w1_M1(sorted_result, :)) \ w1_M1(sorted_result, :)' *inv(diag(cov(sorted_result)))*(true_ratings{a} - mean_rating)';
					elseif a == 5
						P_estimate(a, :) = (lambda(1) * eye(NUM_FACTORS) + w1_M1(sorted_result, :)' * w1_M1(sorted_result, :)) \ w1_M1(sorted_result, :)' * (true_ratings{a} - mean_rating)';
					else
						P_estimate(a, :) = (lambda(2) * eye(NUM_FACTORS) + w1_M1(sorted_result, :)' * w1_M1(sorted_result, :)) \ w1_M1(sorted_result, :)' * (true_ratings{a} - mean_rating)';
					end
					functionval(numel(result{a}), a) = functionval(numel(result{a}), a) + trace(inv(w1_M1(sorted_result, :)'*w1_M1(sorted_result, :) + lambda(2)*eye(NUM_FACTORS)));
				elseif a == 10
					temp_candidate_items = result{a};
					for iter=1:num_iter
						
						rng('shuffle');
						result{a} = randsample(temp_candidate_items, num_items-i);
						result{a} = sort(result{a});
						if cont == 1
							true_ratings{a} = true_P*w1_M1(result{a}, :)' + mean_rating;
						else
							true_ratings{a} = cold_data(find(ismember(cold_data(:, 1), result{a}')), 3)';
						end

						P_estimate(a, :) = (lambda(1) * eye(NUM_FACTORS) + w1_M1(result{a}, :)' * w1_M1(result{a}, :)) \ w1_M1(result{a}, :)' * (true_ratings{a} - mean_rating)';
						
						functionval(numel(result{a}), a) = functionval(numel(result{a}), a) + trace(inv(w1_M1(result{a}, :)'*w1_M1(result{a}, :) + lambda(1)*eye(NUM_FACTORS)));
						
						est_rating = P_estimate(a, :)*w1_M1(test_items, :)' + mean_rating;
						est_rating(est_rating > MAX_RATING) = MAX_RATING;
						est_rating(est_rating < MIN_RATING) = MIN_RATING;

						n = sqrt(sum((est_rating - test_ratings).^2)/numel(test_items));
						RMSE(numel(result{a}), a) = RMSE(numel(result{a}), a) + n;
						P_error(numel(result{a}), a) = P_error(numel(result{a}), a) + norm(P_estimate(a, :) - w1_P1(cold_subset(user), :));
						training_error(numel(result{a}), a) = training_error(numel(result{a}), a) + sqrt(sum(((P_estimate(a, :)*w1_M1(result{a}, :)' + mean_rating) - true_ratings{a}).^2)/numel(result{a}));
					end
				%%% Anti backward
				elseif a == 9
					temp_candidate_items = result{a};
					tic
					result{a} = greedySelect(1, temp_candidate_items, w1_M1, numel(temp_candidate_items) - num_items+i, cov(temp_candidate_items), test);
					timeElapsed = toc;
					result{a} = setdiff(temp_candidate_items, result{a});

					if cont == 1
						true_ratings{a} = true_P*w1_M1(result{a}, :)' + mean_rating;
					else
						true_ratings{a} = cold_data(find(ismember(cold_data(:, 1), result{a}')), 3)';
					end
					
					P_estimate(a, :) = (lambda(3)* eye(NUM_FACTORS) + w1_M1(result{a}, :)'* inv(diag(cov(result{a})))* w1_M1(result{a}, :)) \ w1_M1(result{a}, :)' *inv(diag(cov(result{a})))*(true_ratings{a} - mean_rating)';
					
					functionval(numel(result{a}), a) = functionval(numel(result{a}), a) + trace(inv(w1_M1(result{a}, :)'*inv(diag(cov(result{a})))*w1_M1(result{a}, :) + lambda(3)*eye(NUM_FACTORS)));
					P_estimate(a, :);	
				end
			end
		end
		
		% Get errors for the last iteration
		for a=algos
			if a~=10 % All algos except random
			est_rating = P_estimate(a, :)*w1_M1(test_items, :)' + mean_rating;
			est_rating(est_rating > MAX_RATING) = MAX_RATING;
			est_rating(est_rating < MIN_RATING) = MIN_RATING;
			est_rating = est_rating;
			n = sqrt(sum((est_rating - test_ratings).^2)/numel(test_items));
			user_size_accuracy(user, numel(result{a})) = n;
			RMSE(num_items-i, a) = RMSE(num_items-i, a) + n;
			P_error(num_items-i, a) = P_error(num_items-i, a) + norm(P_estimate(a, :) - w1_P1(cold_subset(user), :));
			training_error(numel(result{a}), a) = training_error(numel(result{a}), a) + sqrt(sum(((P_estimate(a, :)*w1_M1(result{a}, :)' + mean_rating) - true_ratings{a}).^2)/numel(result{a}));
			end
		end
	end
%end
endTime = toc(runTime);

%%% Construct filename for graph plot
name = '';
if dataset == 1
	name = strcat(name, 'netflix_');
elseif dataset == 2
	name = strcat(name, 'jester_');
elseif dataset == 3
	name = strcat(name, 'ml1m_');
elseif dataset == 4
	name = strcat(name, 'ml100k_');
elseif dataset == 5
	name = strcat(name, 'ml20m_');
elseif dataset == 6
	name = strcat(name, 'epi_');
end
figname = strcat(name, num2str(BUDGET), 'users_', num2str(num_iter), 'iter_', num2str(NUM_FACTORS), 'dim_', num2str(num_items), 'items');
if cont == 1
	figname = strcat(figname, '_cont');
end

%%%%%%%%%%% GENERATE PLOT GRAPHS %%%%%%%%%

%%% Plotting RMSE versus number of items selected for all algorithms
fidxs = unique(fidxs);
h = figure;
hold on;
colors = {'h-', 'ro-', 'kd-', 'm-', 'k-', 'c+--', 'y', 'g*:', 'ko--', 'rd--', 'ko:', 'gd-', 'yo-', 'bo-'};
for a=algos
	if a ~= 10
		xaxis = [1:num_items];
		plot(xaxis(fidxs), RMSE(fidxs, a)'/(BUDGET), colors{a}, 'LineWidth', 2)
	else
		xaxis = [1:num_items];
		plot(xaxis(fidxs), RMSE(fidxs, a)'/(num_iter*BUDGET), colors{a}, 'LineWidth', 2)
	end
end

hold off;
xlabel('Number of movies rated')
ylabel('RMSE')
title(strcat('Average number of candidate items per user: ', num2str(avg_per_user/BUDGET)));
algonames = {'ABG2', 'ABG', 'AFG', 'AFG2', 'HV', 'BG2', 'BG', 'FG', 'Anti BG', 'RS', 'FG2', 'Ent0', 'Ent', 'P'};
legend(algonames{algos});

csvwrite(strcat(figname, '.csv'), RMSE)
saveas(h, figname, 'fig');
saveas(h, figname, 'png');

%%% Plotting error in user profile determination versus number of items selected for all algorithms
g = figure;
hold on;
for a=algos
	if a ~= 10
		fidxs = find(P_error(:, a)~=0);
		xaxis = [1:num_items];
		plot(xaxis(fidxs), P_error(fidxs, a)'/(BUDGET), colors{a}, 'LineWidth', 2)
	else
		fidxs = find(P_error(:, a)~=0);
		xaxis = [1:num_items];
		plot(xaxis(fidxs), P_error(fidxs, a)'/(num_iter*BUDGET), colors{a}, 'LineWidth', 2)
	end
end
xlabel('Number of movies rated')
ylabel('User profile error')
legend(algonames{algos});
hold off;
figname = strcat(name, 'PuError_', num2str(BUDGET), 'users_', num2str(num_iter), 'iter_', num2str(NUM_FACTORS), 'dim_', num2str(num_items), 'items');
if cont == 1
	figname = strcat(figname, '_cont');
end
csvwrite(strcat(figname, '.csv'), P_error)
saveas(g, figname, 'fig');
saveas(g, figname, 'png');

%%% Plotting runtime versus number of items selected for all algorithms
fgtime(fgtime == 0) = NaN;
fidxs = ~isnan(fgtime(:, 1));

xaxis = [1:num_items];
f = figure;
hold on;
plot(xaxis(fidxs), bgtime(fidxs, 2)'/(BUDGET), 'ko-', 'LineWidth', 3)
plot(xaxis(fidxs), bgtime(fidxs, 1)'/(BUDGET), 'k-', 'LineWidth', 3)
plot(xaxis(fidxs), bg2time(fidxs, 2)'/(BUDGET), 'go-', 'LineWidth', 3)
plot(xaxis(fidxs), bg2time(fidxs, 1)'/(BUDGET), 'g-', 'LineWidth', 3)
plot(xaxis(fidxs), fgtime(fidxs, 2)'/(BUDGET),  'bo-', 'LineWidth', 3)
plot(xaxis(fidxs), fgtime(fidxs, 1)'/(BUDGET), 'b-', 'LineWidth', 3)
plot(xaxis(fidxs), fg2time(fidxs, 2)'/(BUDGET), 'ro', 'LineWidth', 3)
plot(xaxis(fidxs), fg2time(fidxs, 1)'/(BUDGET), 'r--', 'LineWidth', 3)

legend('BG', 'ABG', 'BG2', 'ABG2', 'FG', 'AFG', 'FG2', 'AFG2');
hold off;
xlabel('Number of movies rated')
ylabel('Time in seconds')
figname = strcat(name, 'time_', num2str(BUDGET), 'users_', num2str(NUM_FACTORS), 'dim_', num2str(num_items), 'items');
if cont == 1
	figname = strcat(figname, '_cont');
end
csvwrite(strcat(figname, '_bgtime.csv'), bgtime)
csvwrite(strcat(figname, '_bg2time.csv'), bg2time)
csvwrite(strcat(figname, '_fgtime.csv'), fgtime)
csvwrite(strcat(figname, '_fg2time.csv'), fg2time)
saveas(f, figname, 'fig');
saveas(f, figname, 'png');