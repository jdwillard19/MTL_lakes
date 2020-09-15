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
# mendota_id = "13293262"
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
        glm_path = raw_dir+"nhd_"+name+"_temperatures.feather"
        geo_path = raw_dir+"nhd_"+name+"_geometry.csv"
        glm = pd.read_feather(glm_path)
        geo = np.genfromtxt(geo_path, delimiter=',', usecols=1)
        surf_area = geo[1]
        n_depths = glm.values.shape[1] - 2
        max_depth = 0.5*(n_depths-1)
        csv.append(name+","+str(max_depth)+","+str(surf_area))


with open('max_depths.csv','a') as file:
    for line in csv:
        file.write(line)
        file.write('\n')
