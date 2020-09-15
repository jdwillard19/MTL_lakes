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

metadata = pd.read_csv("../../../metadata/lake_metadata_wNew.csv")
ids = metadata['nhd_id']
metadata.set_index('nhd_id', inplace=True)
glm_rmse = pd.read_csv('./glm_rmse.csv', header=None, names=['id','rmse'])
for lake in ids:
    nid = lake
    if nid == '120018008' or nid == '120020307' or nid == '120020636' or nid == '32671150' or nid =='58125241'or nid=='120020800' or nid=='91598525':
    	continue
    if math.isnan(metadata.loc[lake].glm_uncal_rmse):
        rmse = glm_rmse[glm_rmse['id'] == lake].rmse.values[0]
        metadata.loc[lake, 'glm_uncal_rmse'] = rmse

metadata.reset_index(inplace=True)

metadata.to_feather("../../../metadata/lake_metadata_wNew2.csv")

