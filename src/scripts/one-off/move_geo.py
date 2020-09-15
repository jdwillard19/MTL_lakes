import pandas as pd
import feather
import numpy as np
import os
import sys
import re
import math
import shutil
from scipy import interpolate
#################################
import pdb##################################################
# (Jared) Nov 2019 - move files
###################################################################################

metadata = pd.read_csv("../../../metadata/lake_metadata.csv")
lakenames = np.array([str(i) for i in metadata.iloc[:,0].values])# to loop through all lakes
for name in lakenames:
	shutil.copyfile("../../../data/raw/figure3_revised/geometry_data/nhd_"+name+"_geometry.csv", "../../../data/processed/lake_data/"+name+"/geometry")