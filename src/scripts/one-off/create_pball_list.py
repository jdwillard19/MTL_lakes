import pandas as pd
import pdb
import numpy as np
import sys
import os
import re
import glob


myPath = '../../../data/raw/usgs_zips/glm/'
listfiles = glob.glob1(myPath,"pball*temperatures.csv")
lst = np.array([re.search('pball_nhdhr_(.*)_temperatures.csv', file).group(1) for file in listfiles])
pdb.set_trace()
csv = []
for el in lst:
	csv.append(el)
with open("../../../metadata/pball_site_ids.csv",'w') as file:
    # print("saving to ../../results/transfer_learning/target_"+target_id+"/resultsPGRNNbasic6")
    for line in csv:
        file.write(line)
        file.write('\n')



