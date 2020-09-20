import os
import re
import pandas as pd
import pdb
import numpy as np
#######################################
# Jan 2019
# Jared - this script creates a bunch of jobs to submit to msi in one script
#######################################


n_lakes = 0
ids = pd.read_csv('../../metadata/pball_site_ids.csv', header=None)
ids = ids[0].values
glm_all_f = pd.read_csv("../../results/glm_transfer/RMSE_transfer_glm_pball.csv")
train_lakes = [re.search('nhdhr_(.*)', x).group(1) for x in np.unique(glm_all_f['target_id'].values)]
n_lakes = len(train_lakes)
test_lakes = ids[~np.isin(ids, train_lakes)]
ids = test_lakes
# ids = ids.values
qsub = ""
# for name in ids.values.flatten():
for name in ids:
    #for each unique lake
    print(name)
    l = name
    m = re.search('{(.+)}', name)
    l2 = name
    if m:
        l2 = m.group(1)
    header = "#!/bin/bash -l\n#PBS -l walltime=23:59:00,nodes=1:ppn=24:gpus=2,mem=16gb \n#PBS -m abe \n#PBS -N %s_pgml_sparse \n#PBS -o %s_pgml_sparse.stdout \n#PBS -q k40 \n"%(l2,l2)
    script = "source takeme_source.sh\n"
    script2 = "source activate mtl_env"
    script3 = "python train_PGDL_custom_sparse.py %s"%(l)
    all= "\n".join([header,script,script2,script3])
    qsub = "\n".join(["qsub job_%s_pgml_sparse.sh"%(l),qsub])
    with open('./jobs/job_{}_pgml_sparse.sh'.format(l), 'w') as output:
        output.write(all)


with open('./jobs/qsub_script_pgml_sparse.sh', 'w') as output2:
    output2.write(qsub)
