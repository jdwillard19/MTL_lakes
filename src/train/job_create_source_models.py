import os
import re
import pandas as pd
import pdb
import numpy as np
#######################################
# Jan 2019
# Jared - this script creates 145 source model creation jobs to submit to msi in one script
# (note: takeme_source.sh must be custom made on users home directory on cluster for this to work: example script 
#        `
#          #!/bin/bash
#         source activate mtl_env
#         cd research/MTL/src/train
#         `  
#######################################


n_lakes = 0
glm_all_f = pd.read_csv("../../results/glm_transfer/RMSE_transfer_glm_pball.csv")
train_lakes = [re.search('nhdhr_(.*)', x).group(1) for x in np.unique(glm_all_f['target_id'].values)]
ids = pd.read_csv('../../metadata/pball_site_ids.csv')
ids = train_lakes
# ids = ids.values
qsub = ""
# for name in ids.values.flatten():
ct = 0
for name in ids:
    ct += 1
    #for each unique lake
    print(name)
    l = name
    m = re.search('{(.+)}', name)
    l2 = name
    if m:
        l2 = m.group(1)
    # if not os.path.exists("../../../models/single_lake_models/"+name+"/PGRNN_basic_normAll_pball"): 
    header = "#!/bin/bash -l\n#PBS -l walltime=23:59:00,nodes=1:ppn=24:gpus=2,mem=16gb \n#PBS -m abe \n#PBS -N %s_pgml_source \n#PBS -o ./jobs/%s_pgml_source.stdout \n#PBS -q k40 \n"%(l2,l2)
    script = "source takeme_source.sh\n" #cd to directory with training script
    script2 = "source activate mtl_env"
    script3 = "python train_source_model.py %s"%(l)
    # script3 = "python singleModel_customSparse.py %s"%(l)
    all= "\n".join([header,script,script2,script3])
    qsub = "\n".join(["qsub job_%s_pgml_source.sh"%(l),qsub])
    with open('./jobs/job_{}_pgml_source.sh'.format(l), 'w') as output:
        output.write(all)


with open('./jobs/qsub_script_pgml_source.sh', 'w') as output2:
    output2.write(qsub)

print(ct, " jobs created")