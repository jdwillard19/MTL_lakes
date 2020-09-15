import os
import re
import sys
import numpy as np
#######################################
# Jan 2019
# Jared - this script creates a bunch of jobs to submit to msi in one script
#######################################


raw_dir = '../../../data/raw/figure3' #unprocessed data directory
lnames = set()
n_lakes = 0
csv = []
mendota_id = "13293262"
m_glm_path = raw_dir+"nhd_"+mendota_id+"_temperatures_wEnergy.feather"
m_glm = pd.read_feather(m_glm_path)
for filename in os.listdir(directory):
    #parse lakename from file
    m = re.search(r'^nhd_(\d+)_.*', filename)
    if m is None:
        continue
    name = m.group(1)
    if name not in lnames:
        #for each unique lake
        print(name)