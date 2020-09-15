import re
import sys
import os
import pandas as pd
import numpy as np
import pdb
glm_all_f = pd.read_csv("../../../results/glm_transfer/RMSE_transfer_glm_pball.csv")
train_lakes = [re.search('nhdhr_(.*)', x).group(1) for x in np.unique(glm_all_f['target_id'].values)]


pt1 = re.compile("^pretraining\s+finished\s+in\s+(\d+)\s+epochs")
pt2 = re.compile("^training\s+finished\s+in\s+(\d+)")



n_lakes = len(train_lakes)
pt_eps_pr = np.empty((n_lakes))
tr_eps_pr = np.empty((n_lakes))
pt_eps_pr[:] = np.nan
tr_eps_pr[:] = np.nan
pt_eps_nw = np.empty((n_lakes))
tr_eps_nw = np.empty((n_lakes))
pt_eps_nw[:] = np.nan
tr_eps_nw[:] = np.nan

ct = -1
for i_d in train_lakes:
    print("i_d: ",i_d)
    ct += 1
    if os.path.exists('../manylakes2/jobs/'+str(i_d)+'_pgml_pball.stdout'):
        for i, line in enumerate(open('../manylakes2/jobs/'+str(i_d)+'_pgml_pball.stdout')):
            for match in re.finditer(pt1, line):
                print("pt ep p: ",match.group(1))
                pt_eps_pr[ct] = float(match.group(1))
                break
            for match2 in re.finditer(pt2, line):
                print("tr ep p: ",match2.group(1))
                tr_eps_pr[ct] = float(match2.group(1))
                break
    if ~np.isfinite(pt_eps_pr[ct]):
        for i, line in enumerate(open('../manylakes2/jobs/'+str(i_d)+'_pgml_ge10.stdout')):
            for match in re.finditer(pt1, line):
                print("pt ep p: ",match.group(1))
                pt_eps_pr[ct] = float(match.group(1))
                break
    if ~np.isfinite(tr_eps_pr[ct]):
        for i, line in enumerate(open('../manylakes2/jobs/'+str(i_d)+'_pgml_ge10.stdout')):
            for match in re.finditer(pt2, line):
                print("tr ep p: ",match.group(1))
                pt_eps_pr[ct] = float(match.group(1))
                break
    for i, line in enumerate(open('../manylakes2/jobs/'+str(i_d)+'_pgml_source.stdout')):
        for match in re.finditer(pt1, line):
            print("pt ep p: ",match.group(1))
            pt_eps_nw[ct] = float(match.group(1))
            break
        for match in re.finditer(pt2, line):
            print("tr ep n: ",match.group(1))
            tr_eps_nw[ct] = float(match.group(1))
            break

pdb.set_trace()
tr_eps_pr = tr_eps_pr[np.where(np.isfinite(pt_eps_pr))]
tr_eps_nw = tr_eps_nw[np.where(np.isfinite(pt_eps_pr))]
pt_eps_nw = pt_eps_nw[np.where(np.isfinite(pt_eps_pr))]
pt_eps_pr = pt_eps_pr[np.where(np.isfinite(pt_eps_pr))]

tr_eps_pr = tr_eps_pr[np.where(np.isfinite(pt_eps_nw))]
tr_eps_nw = tr_eps_nw[np.where(np.isfinite(pt_eps_nw))]
pt_eps_pr = pt_eps_pr[np.where(np.isfinite(pt_eps_nw))]
pt_eps_nw = pt_eps_nw[np.where(np.isfinite(pt_eps_nw))]

tr_eps_nw = tr_eps_nw[np.where(np.isfinite(tr_eps_pr))]
pt_eps_pr = pt_eps_pr[np.where(np.isfinite(tr_eps_pr))]
pt_eps_nw = pt_eps_nw[np.where(np.isfinite(tr_eps_pr))]
tr_eps_pr = tr_eps_pr[np.where(np.isfinite(tr_eps_pr))]

pt_eps_pr = pt_eps_pr[np.where(np.isfinite(tr_eps_nw))]
pt_eps_nw = pt_eps_nw[np.where(np.isfinite(tr_eps_nw))]
tr_eps_pr = tr_eps_pr[np.where(np.isfinite(tr_eps_nw))]
tr_eps_nw = tr_eps_nw[np.where(np.isfinite(tr_eps_nw))]
print(tr_eps_nw.shape)


print("old pretain (med/lower/upper)\n",np.median(pt_eps_pr),"\n",np.quantile(pt_eps_pr,.25),"\n",np.quantile(pt_eps_pr,.75))
print("old train (med/lower/upper)\n",np.median(tr_eps_pr),"\n",np.quantile(tr_eps_pr,.25),"\n",np.quantile(tr_eps_pr,.75))

print("new pretain (med/lower/upper)\n",np.median(pt_eps_nw),"\n",np.quantile(pt_eps_nw,.25),"\n",np.quantile(pt_eps_nw,.75))
print("new train (med/lower/upper)\n",np.median(tr_eps_nw),"\n",np.quantile(tr_eps_nw,.25),"\n",np.quantile(tr_eps_nw,.75))

