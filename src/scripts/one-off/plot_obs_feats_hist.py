import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pdb

data = pd.read_csv("./obs_and_feats_hist.csv", header=None)
x = data.iloc[:,0]
y = data.iloc[:,1]

fig, axs = plt.subplots(1, 2, tight_layout=True)

# We can set the number of bins with the `bins` kwarg
# n_bins1 = [0,50,100,150,200,400,800,2000]
# n_bins2 = [0,50,100,150,200,400,800,2000]
n_bins = 40
axs[0].hist(x, bins=n_bins)
axs[1].hist(y, bins=n_bins)
axs[0].set_ylabel("Frequency")
axs[0].set_xlabel("Number of Temperature Measurements")
axs[1].set_xlabel("Number of Sampling Dates")
plt.show()
