import os
import re
import pandas as pd
import pdb
#######################################
# Jan 2019
# Jared - this script creates a bunch of jobs to submit to msi in one script
#######################################


directory = '../../../data/raw/figure3' #unprocessed data directory
lnames = set()
n_lakes = 0
ids = pd.read_csv("../../../metadata/sites_moreThan10ProfilesWithGLM.csv")
ids = ids['site_id'].values
qsub = ""
ct = 0
full_ct = 0
part_ct = 0
both_ct = 0
feat_ct = 0
new_ids = []
for name in ids:
    #for each unique lake
    lnames.add(name)
    l = name
    m = re.search('{(.+)}', name)
    l2 = name
    if m:
        l2 = m.group(1)
    full_ex = os.path.exists("../../../models/single_lake_models/"+name+"/PGRNN_basic_normAllGr10")
    part_ex = os.path.exists("../../../models/single_lake_models/"+name+"/PGRNN_basic_normAllGr10_partial")
    feat_ex = os.path.exists("../../../data/processed/lake_data/"+name+"/features.npy")
    if not feat_ex:
        feat_ct += 1

    if not full_ex: 
        # print(l, " missing full")
        full_ct += 1

    if not part_ex: 
        # print(l, " missing partial")
        part_ct += 1
        

    if not full_ex and not part_ex:
        both_ct += 1
    if full_ex and part_ex and feat_ex:
        new_ids.append(name)

pdb.set_trace()
new_ids = pd.DataFrame(new_ids, names=['id']) 
new_ids.to_csv("../../../metadata/sites_moreThan10ProfilesWithGLM2.csv")
print("full, ",full_ct, " / ", len(ids))
print("part, ", part_ct, " / ", len(ids))
print("both, ", both_ct, " / ", len(ids))
print("feat, ", feat_ct, " / ", len(ids))
