# optimal-interview-design
Steps to run the experiments

## Pre-processing and splitting into train and test
For some odd reason, MATLAB reads .mat files much faster than .csv files. Python on the other hand, reads CSV files very fast.
netflixToMatRand3.py reads in a CSV file, and preps it for further processing by MATLAB by converting it to .mat. It also does some pre-processing steps, such as randomly splitting the pool of users into cold (test set) and warm (training set), and separating the ratings by each group.

To run,
1. Fill in the directory where the raw CSV files are stored.
2. Uncomment the metadata for the dataset you're working with, and comment out the rest.
3. If working with a dataset not described in the file, please add it using the same variable names as for the other datasets.

## Running the PMF model
Follow instructions given in https://github.com/sampoorna/probabilistic-matrix-factorization to train a PMF model on only ratings given by warm users (randomly determined in the previous step). Note that for these experiments, we only ran PMF with the diminishing step-size adaptation, and not Bayesian PMF.

## Computing user profiles
Before running the experiments, we need to determine what the ground truth for our test set (cold users) will be. This is what we will compare our predicted user profiles against, to measure profile error. However, as they are cold users, they cannot be part of the dataset when we factorize the matrix of ratings.

To get around this problem, we assume that each cold user is independent and that the user latent vector for each can be represented as a linear combination of the item vectors (obtained from the step above) of the items that we have ratings on record for. This has been shown to 

To run,
1. Fill in the directory where the model files (for the model trained on only ratings given by warm users) are stored.
2. Set `dataset = x` for the dataset being used.
3. If working with a dataset not described in the file, please add it using the same variable names as for the other datasets.

The output of this script is a matrix of cold user vectors that we will use as ground truth in the next step.

## Running item selection algorithms
This script has various parts to it.
1. Loads processed training and test data.
2. Runs various item selection algorithms and baselines.
3. Computes test and training error (as RMSE).
4. Computes run time.
5. Plots run time and test error.

To run,
1. Fill in the directory where the model files (for the model trained on only ratings given by warm users) are stored.
2. Set `dataset = x` for the dataset being used.
3. There are other options as well, such as,
    - `retrain`: set it to 1 when covariance and baseline results need to be computed again, else 0
    - `reload`: set it to 1 when variables need to be loaded in the MATLAB workspace, else 0
    - `cont`: set it to 1 to validate against P*Q (ideal setting) instead of actual ratings in the database (real setting), 0 otherwise
4. Some parameters to set are
    - `NUM_FACTORS`: Number of latent dimensions or size of latent vector
    - `lambda`: Array of noise or hyper-parameters. They should be small numbers that are added to the the objective funtion to ensure invertibility
    - `num_items`: Maximum number of items that are used to learn the cold user's profile. The experiment setup runs all the algorithms from 1 upto this many items.
    - `BUDGET`: Number of cold users to average results across
    - `num_iter`: Number of iterations to average results of the random baseline across
    - `algos`: Array of numbers corresponding to the algorithms that should be run against each other in a single experiment