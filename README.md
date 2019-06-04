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
Follow instructions given in https://github.com/sampoorna/probabilistic-matrix-factorization to train a PMF model on only ratings given by warm users (randomly determined in the previous step).

## Computing user profiles
Before running the experiments, we need to determine what the ground truth for our test set (cold users) will be. This is what we will compare our predicted user profiles against, to measure profile error. However, as they are cold users, they cannot be part of the dataset when we factorize the matrix of ratings.

To get around this problem, we assume that each cold user is independent and that the user latent vector for each can be represented as a linear combination of the item vectors (obtained from the step above) of the items that we have ratings on record for. This has been shown to 

To run,
1. Fill in the directory where the model files (for the model trained on only ratings given by warm users) are stored.
2. Set `dataset = x` for the dataset being used.
3. If working with a dataset not described in the file, please add it using the same variable names as for the other datasets.

The output of this script is a matrix of cold user vectors that we will use as ground truth in the next step.

## Running item selection algorithms
Run greedyBudget.m
Good luck figuring the code out (I'm serious, if you do, please clean it up). 