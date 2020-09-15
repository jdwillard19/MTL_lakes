import os
import re
import sys
import numpy as np
import pandas as pd
import pdb
import math
#######################################
# Oct 2019
# Jared - calculate if each lake is well mixed or not 
# stratification condition: 1 °C between the shallowest and deepest depths measured for >70% of profiless
#######################################


raw_dir = '../../../data/raw/figure3/' #unprocessed data directory

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

metadata = pd.read_csv("../../../metadata/lake_metadata_wNew.csv")
meta_old = pd.read_csv("../../../metadata/lake_metadata.csv")
ids = metadata['nhd_id']
metadata.set_index('nhd_id', inplace=True)
metadata.columns = [c.replace(' ', '_') for c in metadata.columns]
csv = []
ct = 0
min_sa = 86709.46693
max_sa = 929821943.1
min_d = 5
max_d = 35
for lake in ids:
    nid = lake
    if nid == '120018008' or nid == '120020307' or nid == '120020636' or nid == '32671150' or nid =='58125241'or nid=='120020800' or nid=='91598525':
    	continue

    d = metadata.loc[lake].max_depth
    d_over = d > max_d
    d_under = d < min_d

    sa = metadata.loc[lake].surface_area
    sa_over = sa > max_sa
    sa_under = sa < min_sa
    csv.append(",".join([lake, str(d_over), str(d_under), str(sa_over), str(sa_under)]))


#120017989

with open('metadata_oob.csv','a') as file:
    for line in csv:
        file.write(line)
        file.write('\n')