import os
import re
import sys
import numpy as np
import pandas as pd
#######################################
# July 2019
# Jared - calculate max depth ea lake
#######################################


raw_dir = '../../../data/raw/figure3/' #unprocessed data directory
lnames = set()
n_lakes = 0
csv = []
target_id = "13293262"
m_path = '../manylakes2/labels/'+target_id+'_full_label.feather'
lbl = pd.read_feather(m_path)
# mendota_id = "13293262"

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())
# m_obs_path = raw_dir+"nhd_"+mendota_id+"_test_train.feather"
# start_date = "{:%Y-%m-%d}".format(obs.values[0,1]).encode()
# end_date = "{:%Y-%m-%d}".format(obs.values[-1,1]).encode()
# m_glm = pd.read_feather(m_glm_path)
# lnames.add(mendota_id)
for filename in os.listdir(raw_dir):
    #parse lakename from file
    m = re.search(r'^nhd_(\d+)_.*', filename)
    if m is None:
        continue
    name = m.group(1)
    if name not in lnames:
        lnames.add(name)
        #for each unique lake
        o_path = "../manylakes2/single_models/"+name+"PGRNNoutputOn"+target_id+".feather"
        o = pd.read_feather(o_path)
        oval = np.array(o.values[:-4000,1:-2],dtype=np.float64)
        lval = np.array(lbl.values[:-4000,1:], dtype=np.float64)
        rmse = rmse(oval[~np.isnan(lval)], lval[~np.isnan(lval)])
        print(name, ":",rmse)
        sys.exit()



with open('max_depths.csv','a') as file:
    for line in csv:
        file.write(line)
        file.write('\n')
