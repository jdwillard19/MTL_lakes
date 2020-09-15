import pandas as pd
import numpy as np
import pdb
import re
import os

#use bins to separate into train and test
metadata = pd.read_csv("../../../metadata/lake_metadata_wNew.csv")
ids = metadata['nhd_id']
# metadata.set_index('nhd_id')
metadata.sort_values(by=['surface_area'], inplace=True)
d_bins = [5, 10, 20, 30]
bins = np.empty((5,5), dtype=np.object)
bins[:,0] = [], [], [], [], []
bins[:,1] = [], [], [], [], []
bins[:,2] = [], [], [], [], []
bins[:,3] = [], [], [], [], []
bins[:,4] = [], [], [], [], []
d_inds = np.digitize(metadata['max_depth'], d_bins)
metadata['d_bin'] = d_inds
bin_by_a = np.array_split(metadata, 5)

new_df = pd.DataFrame()
new_df['id'] = metadata['nhd_id']
new_df['isTrain'] = False
new_df['bin'] = -1
new_df['d_bin'] = -1
new_df['a_bin'] = -1
for i in range(len(bin_by_a)):
    df = bin_by_a[i]
    for index, row in df.iterrows():
        nid = 'nhdhd_'+row['nhd_id']
        if nid == 'nhdhr_120018008' or nid == 'nhdhr_120020307' or nid == 'nhdhr_120020636' or nid == 'nhdhr_32671150' or nid =='nhdhr_58125241':
            print("discarded ", index)
            continue
        d_ind = row['d_bin']
        a_ind = i
        bins[d_ind, a_ind].append(row['nhd_id'])

dropInd0 = new_df[ new_df['id'] == '120018008' ].index
dropInd1 = new_df[ new_df['id'] == '120020307' ].index
dropInd2 = new_df[ new_df['id'] == '120020636' ].index
dropInd3 = new_df[ new_df['id'] == '32671150' ].index
dropInd4 = new_df[ new_df['id'] == '58125241' ].index
# new_df.drop([dropInd0, dropInd1, dropInd2, dropInd3, dropInd4] , inplace=True)
new_df.drop([dropInd0[0], dropInd1[0], dropInd2[0], dropInd3[0], dropInd4[0]] , inplace=True)
flat_bins = bins.flatten()
for i in range(flat_bins.shape[0]):
    bn = i + 1
    #from each bin take 2/3rd for training 1/3rd for test
    cutoff_ind = int(np.round(2*len(flat_bins[i])/3))
    if cutoff_ind > 0:
        train_ids = np.random.choice(flat_bins[i], size=cutoff_ind, replace=False)
        ids = flat_bins[i]
        for idnum in train_ids.tolist():
            new_df.loc[new_df['id'] == idnum, 'isTrain'] = True
        for idnum in ids:
            new_df.loc[new_df['id'] == idnum, 'bin'] = bn

            #get d and a bin indicis
            for d_ind in range(5):
                for a_ind in range(5):
                    if idnum in bins[d_ind, a_ind]:
                        new_df.loc[new_df['id'] == idnum, 'd_bin'] = d_ind
                        new_df.loc[new_df['id'] == idnum, 'a_bin'] = a_ind

new_df.reset_index(inplace=True)
new_df.to_feather("../../../data/processed/lake_splits/trainTestNewLakes.feather")


# for lake in ids:
#     nid = 'nhdhd_'+lake
#     if nid == 'nhdhr_120018008' or nid == 'nhdhr_120020307' or nid == 'nhdhr_120020636' or nid == 'nhdhr_32671150' or nid =='nhdhr_58125241':
#         continue
#     d = metadata.loc[str(lake)].max_depth





    

