#!/usr/env python3

import pandas as pd
data = pd.read_csv('data.csv', header=None)

# make sure the last file having the remaining rows
num_rows = data.shape[0]

# get the number of rows in each file I change this from 99 to 49
num_rows_each_file = int(num_rows / 49)

# split the dataset into 49 files and the last file having the remaining rows
for i in range(49):
    # get the rows for each file
    rows = data.iloc[i * num_rows_each_file: (i + 1) * num_rows_each_file, :]

    # save the file
    rows.to_csv('sfile_' + str(i+1) + '.csv', index=False, header=None)

# the last file having the remaining rows
rows = data.iloc[49 * num_rows_each_file:, :]
rows.to_csv('sfile_50.csv', index=False, header=None)
