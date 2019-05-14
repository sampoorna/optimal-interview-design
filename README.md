# optimal-interview-design
Steps to run the experiments

## netflixToMatRand3.py
For some odd reason, MATLAB reads .mat files much faster than .csv files. Python on the other hand, reads CSV files very fast.
This script reads in a CSV file, and preps it for further processing by MATLAB by converting it to .mat. It also does some pre-processing steps, such as randomly splitting the pool of users into cold and warm, and separating the ratings by each group.

To run,
1. Fill in the directory where the raw CSV files are stored.
2. Uncomment the metadata for the dataset you're working with, and comment out the rest.
3. If working with a dataset not described in the file, please add it using the same variable names as used for the other datasets.