import os
import re
import pandas as pd
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
for name in ids:
    #for each unique lake
    lnames.add(name)
    l = name
    m = re.search('{(.+)}', name)
    l2 = name
    if m:
        l2 = m.group(1)

    if not os.path.exists("../../../models/single_lake_models/"+name+"/PGRNN_basic_normAllGr10"): 
        header = "#!/bin/bash -l\n#PBS -l walltime=14:00:00,nodes=1:ppn=24:gpus=2,mem=16gb \n#PBS -m abe \n#PBS -N %s_pgml_ge10 \n#PBS -o %s_pgml_ge10.stdout \n#PBS -q k40 \n"%(l2,l2)
        script = "source job_takeme.sh\n"
        script2 = "source activate pytorch4"
        script3 = "python realSingleModelNorm2_ge10.py %s"%(l)
        all= "\n".join([header,script,script2,script3])
        qsub = "\n".join(["qsub job_%s_pgml_ge10_2.sh"%(l),qsub])
        with open('./jobs/job_{}_pgml_ge10_2.sh'.format(l), 'w') as output:
            output.write(all)

    # if not os.path.exists("../../../models/single_lake_models/"+name+"/PGRNN_basic_normAllGr10_partial"): 
    #     header = "#!/bin/bash -l\n#PBS -l walltime=14:00:00,nodes=1:ppn=24:gpus=2,mem=16gb \n#PBS -m abe \n#PBS -N %s_pgml_ge10 \n#PBS -o %s_pgml_ge10.stdout \n#PBS -q k40 \n"%(l2,l2)
    #     script = "source job_takeme.sh\n"
    #     script2 = "source activate pytorch4"
    #     script3 = "python realSingleModelNorm2_ge10_partial.py %s"%(l)
    #     all= "\n".join([header,script,script2,script3])
    #     qsub = "\n".join(["qsub job_%s_pgml_ge10_partial_2.sh"%(l),qsub])
    #     with open('./jobs/job_{}_pgml_ge10_partial_2.sh'.format(l), 'w') as output:
    #         output.write(all)


with open('./jobs/qsub_script_pgml_ge10_3.sh', 'w') as output2:
    output2.write(qsub)
