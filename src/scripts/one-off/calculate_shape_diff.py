import torch
import pandas as pd
import numpy as np
import re
import feather
import os
import sys
from scipy import interpolate

sys.path.append('../../data')
from pytorch_data_operations import calculate_energy, transformTempToDensity, getHypsography, buildLakeDataForRNNPretrain

raw_dir = '../../../data/raw/figure3/' #unprocessed data directory
proc_dir = '../../../data/processed/WRR_69Lake/'

lnames = set()
n_features = 7
n_lakes = 0
csv = []
loi = '1099282'
lnames.add(loi)
############################################################################################################################
# July 2019 - script to add energy information to GLM temperature feather files
#####################################################################################################################
for filename in os.listdir(raw_dir):
    #parse lakename from file
    m = re.search(r'^nhd_(\d+)_.*', filename)
    if m is None:
        continue
    name = m.group(1)
    if name not in lnames:
        print(name)
        #for each unique lake
        data_dir = "../../data/processed/WRR_69Lake/"+name+"/"

        lnames.add(name)
        glm_path = raw_dir+"nhd_"+name+"_temperatures.feather"
        glm_path2 = raw_dir+"nhd_"+name+"_temperatures_wEnergy.feather"
        geo_path = raw_dir+"nhd_"+name+"_geometry.csv"
        glm = pd.read_feather(raw_dir+"nhd_"+name+"_temperatures.feather")
        temps = np.array(glm.values[:,1:-1], dtype=np.float32)
        temps = np.rot90(temps, k=3).copy()
        (_, _, _, _, hypsography) = buildLakeDataForRNNPretrain(name, data_dir, 200, 8,
                                       win_shift= 100, begin_loss_ind=20,
                                       excludeTest=False)
        print(hypsography)

        #interpolate missing data
        n_depths = temps.shape[0]
        max_depth = 0.5*(n_depths-1)
        depths = np.arange(0, max_depth+0.5, 0.5)
        if np.isnan(np.sum(temps)):
            # print("Warning: there is missing data in glm output")
            for i in range(temps.shape[0]):
                #for each depth
                for t in range(temps.shape[1]):
                    if np.isnan(temps[i,t]):
                        x = depths[i]
                        xp = depths[0:(i-1)]
                        yp = temps[0:(i-1),t]
                        f = interpolate.interp1d(xp, yp,  fill_value="extrapolate")
                        temps[i,t] = f(x) #interp_temp

            assert not np.isnan(np.sum(temps))
        torch_temps = torch.from_numpy(temps)
        torch_hyps = torch.from_numpy(hypsography).float()[0]
        energies = calculate_energy(torch_temps, torch_hyps, False)
        new_col = pd.DataFrame({'energy':energies})
        glm = pd.concat([glm, new_col], axis = 1)
        energies_norm = (energies - energies[~np.isnan(energies)].mean()) / energies[~np.isnan(energies)].std()
        new_col2 = pd.DataFrame({'energy_norm':energies_norm})
        glm = pd.concat([glm, new_col2], axis = 1)
        n_depths = temps.shape[1]
        # print(np.sum([0,:]*hypsography))
        feather.write_dataframe(glm, glm_path2)
