import pandas as pd

import numpy as np

import scipy.stats as st

for i in range(7,20):

    df = pd.read_csv(f'./uniform_comparison_{i}.csv',header=None)

    k = int(np.floor( (i-2)/2))

    temp = [[] for x in range(k)]

    for ind, row in df.iterrows():

        for x in range(k):

            temp[x].append(eval(row[x])[-1])

    print(i,[len(temp[y]) for y in range(k)])

    print(i, 'Mean', [np.mean(temp[y]) for y in range(k)])

    print(i,'Standard Deviation', [np.std(temp[y]) for y in range(k)])

    print(i,'Skewness',[st.skew(temp[y]) for y in range(k)])
