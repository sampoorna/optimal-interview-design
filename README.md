# optimal-interview-design
Steps to run the experiments

## Pre-processing and splitting into train and test
For some odd reason, MATLAB reads .mat files much faster than .csv files. Python on the other hand, reads CSV files very fast.
netflixToMatRand3.py reads in a CSV file, and preps it for further processing by MATLAB by converting it to .mat. It also does some pre-processing steps, such as randomly splitting the pool of users into cold (test set) and warm (training set), and separating the ratings by each group.

To run,
1. Fill in the directory where the raw CSV files are stored.
2. Uncomment the metadata for the dataset you're working with, and comment out the rest.
3. If working with a dataset not described in the file, please add it using the same variable names as used for the other datasets.

## Running the PMF model
Follow instructions given in https://github.com/sampoorna/probabilistic-matrix-factorization to train a PMF model on only ratings given by warm users (randomly determined in the previous step).

## Running item selection algorithms
Run greedyBudget.m
Good luck figuring the code out (I'm serious, if you do, please clean it up). 