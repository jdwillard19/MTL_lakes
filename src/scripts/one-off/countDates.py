import pandas as pd
import feather
import numpy as np
import os
import sys
import re
import shutil
from scipy import interpolate
import csv

###################################################################################
# (Jared) June 2019 - Count number of training and test dates in WRR Figure 3 experiment
###################################################################################
directory = '../../../data/raw/figure3' #unprocessed data directory
lnames = set()
n_lakes = 0

with open('counts.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=' ',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
    for filename in os.listdir(directory):
        #parse lakename from file
        m = re.search(r'^nhd_(\d+)_.*', filename)
        if m is None:
            continue
        name = m.group(1)
        print(name)
        if name not in lnames:
            #for each unique lake
            lnames.add(name)
            test_obs = pd.read_feather("../../../data/raw/figure3_revised/test data/nhd_"+name+"_test_all_profiles.feather")
            vals = test_obs.val
            count = np.unique(vals[:,1]).shape[0]
            writer.writerow([name, str(count)])