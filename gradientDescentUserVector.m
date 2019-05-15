%%%%%%%%%%% README %%%%%%%%%%%
% Calculate user vectors using gradient descent and ratings. These are used as the ground truth user vectors of the cold users.

NUM_FACTORS = 20;

fprintf('Loading files...\n');
if dataset == 1 %%% Netflix
	origdirec = '../data/netflix/';
	MAX_ITEMS = 17770;
	MAX_USERS = 2649429;
elseif dataset == 2 %%% Jester
	load '../data/rating_matrix_all.mat';
	load '../data/jester_data_withoutrat_randcold.mat';
	load '../data/jester_w1_M1_20.mat';
elseif dataset == 3 %%% MovieLens 1M
	origdirec = '../data/ml-1m/'; 
	MAX_ITEMS = 3952;
	MAX_USERS = 6040;
elseif dataset == 4 %%% MovieLens 100K
	origdirec = '../data/ml-100k/'; 
	MAX_ITEMS = 1682;
	MAX_USERS = 943;
elseif dataset == 5 %%% MovieLens 20M
	origdirec = '../data/recsys/MovieLens/ml-20m/model/';
	MAX_ITEMS = 131262;
	MAX_USERS = 138493;
elseif dataset == 6 %%% Epinions
	origdirec = '../data/recsys/Epinions/model/';
	MAX_ITEMS = 139738;
	MAX_USERS = 49290;
end
load(strcat(origdirec, 'data_withoutrat_randcold2.mat'));
load(strcat(origdirec, 'w1_M1_', num2str(NUM_FACTORS), '.mat'));
fprintf('Loading files...COMPLETE\n');

%%% Go over each user in the cold user dataset
cold_users = unique(cold(:, 2));
mean_rating = mean(warm(:, 3));
cold_user_vectors = sparse(0.25*ones(MAX_USERS, NUM_FACTORS));
options = optimoptions(@fminunc, 'MaxFunEvals', 5000, 'Display','iter');

for cut=1:numel(cold_users')
	cu = cold_users(cut);
	fprintf('Computing for ... %d\n', cu)
	% Get ratings of this user
	cold_user_indices = find(cold(:, 2) == cu);
	rated_item_indices = cold(cold_user_indices, 1);
	cold_ratings = cold(cold_user_indices, 3);
	% Discard if too few, else compute vector by GD
	if numel(cold_ratings >= 30)
		item_vectors = w1_M1(rated_item_indices, :);
		error = @(x)sum((item_vectors*x' + mean_rating - cold_ratings).^2);
		[cold_user_vectors(cu, :), fval] = fminunc(error, cold_user_vectors(cu, :), options);
		fprintf('%f .... \n', fval)
	end
end

% Save variable
w1_P1 = cold_user_vectors;
save(strcat(origdirec, 'gd_w1_P1_', num2str(NUM_FACTORS), '.mat'), 'w1_P1');