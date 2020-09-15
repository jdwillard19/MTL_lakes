import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pdb



transfer_data = pd.read_feather("../../../results/pgml_transfer_stats.feather")
pgml_data = pd.read_feather("../../../results/customSparseResults.feather")


n_profiles = np.array([1,2,5,10,15,20,25,30,35,40,45,50])


med_per_n_prof = np.empty_like((n_profiles))
for n_prof in n_profiles:
	pdb.set_trace()

