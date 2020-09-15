import os
import re
import sys
import numpy as np
import pandas as pd
import pdb
import math
#######################################
# Dec 2019
# Jared - merge trn/tst to one file
#######################################

metadata = pd.read_csv("../../../metadata/lake_metadata_wNew.csv")
ids = metadata['nhd_id']
metadata.set_index('nhd_id', inplace=True)
csv = []
ct = 0
for lake in ids:
    nid = lake
    if nid == '120018008' or nid == '120020307' or nid == '120020636' or nid == '32671150' or nid =='58125241'or nid=='120020800' or nid=='91598525':
    	continue

    print("lake ", lake)

    #load data
    tst = np.load('../../../data/processed/lake_data/'+lake+'/test_b.npy')
    trn = np.load('../../../data/processed/lake_data/'+lake+'/train_b.npy')

    #combine into full data
    na = np.isnan(tst)
    nb = np.isnan(trn)
    tst[na] = 0
    trn[nb] = 0
    tst += trn
    na &= nb
    tst[na] = np.nan
    full = tst
    np.save('../../../data/processed/lake_data/'+lake+'/full_obs.npy', full)


