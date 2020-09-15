import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pdb

obs = pd.read_csv("../../../../../Documents/obs.csv")
obs = obs.values
hist,bin_edges = np.histogram(obs)

plt.hist(obs, bins=15)
plt.ylabel('Frequency');
pdb.set_trace()
# plt.xticks([str(a) for a in np.arange(obs.min(), obs.max()+1, 200.0)])
plt.xticks(np.arange(obs.min()-2, obs.max()+1, 200.0))
plt.title("Histogram of number of observation days per source lake")
plt.show()