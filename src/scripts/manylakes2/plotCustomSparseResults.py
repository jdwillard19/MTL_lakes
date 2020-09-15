import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pdb
from scipy.interpolate import interp1d
import sys
import matplotlib.patheffects as pe
import matplotlib

font = {'family' : 'serif',
        'size'   : 10}  
matplotlib.rc('font', **font)
transfer_data = pd.read_feather("../../../results/pgml_transfer_stats_July2020.feather")
# transfer_data_old = pd.read_feather("../../../results/pgml_transfer_stats_July2020.feather")
pgml_data = pd.read_feather("../../../results/sparseModelResults_July2020.feather")
pgml_data_old = pd.read_feather("../../../results/customSparseResults_old.feather")
# ens_data = pd.read_csv("")
plot_data = pd.read_csv("../bean_plot_data.csv")

lakes = transfer_data['site_id']
cross_per_lake = np.empty_like(lakes.values)
cross_per_lake[:] = np.nan
n_profiles = np.array([1,2,5,10,15,20,25,30,35,40,45,50])


med_per_n_prof = np.empty((n_profiles.shape[0]))
std_per_n_prof = np.empty((n_profiles.shape[0]))
lq_per_n_prof = np.empty((n_profiles.shape[0]))
uq_per_n_prof = np.empty((n_profiles.shape[0]))
mad_per_n_prof = np.empty((n_profiles.shape[0]))
for i, n_prof in enumerate(n_profiles):
  med = pgml_data[str(n_prof)+' obs median'].values
  med_o = pgml_data_old[str(n_prof)+' obs median'].values
  std = pgml_data[str(n_prof)+' obs std'].values
  mad = pgml_data[str(n_prof)+' obs mad'].values
  std = std[med != 0]
  mad = mad[med != 0]
  med = med[med != 0]
  med_o = med_o[med_o != 0]
  med_per_n_prof[i] = np.median(med[np.isfinite(med)])
  print(n_prof, " median (new/old): (", np.median(med[np.isfinite(med)])," / ", np.median(med_o[np.isfinite(med_o)]),")")
  std_per_n_prof[i] = np.median(std[np.isfinite(std)])
  mad_per_n_prof[i] = np.median(std[np.isfinite(mad)])
  lq_per_n_prof[i] = np.quantile(med[np.isfinite(med)],.25)
  uq_per_n_prof[i] = np.quantile(med[np.isfinite(med)],.75)

pdb.set_trace()
median_selected = transfer_data['best_pred'].median()
worst_selected = transfer_data['highest'].median()
best_selected = transfer_data['lowest'].median()


#calculate crossover point per lake
# for site_ct, site_id in enumerate(lakes.values):
#   tdata = transfer_data[transfer_data['site_id'] == site_id]

#   spdata = pgml_data[pgml_data['site_id'] == site_id]
#   x = n_profiles
#   y = np.array([spdata[str(n_prof)+' obs median'].values[0] for n_prof in n_profiles])
#   y[y<0.1] = np.nan
#   x = x[np.isfinite(y)]
#   y = y[np.isfinite(y)]
#   f = interp1d(x, y)
#   xs = np.arange(1,x[-1],.1)
#   ys = np.empty_like(xs)
#   ys[:] = np.nan

#   for ct, i in enumerate(xs):
#     ys[ct] = f(i)
#   plt.plot(xs,ys)
#   plt.show()
#   ys = ys - tdata['best_pred'].values[0]
#   cross_per_lake[site_ct] = np.argmin(np.abs(ys))

# plt.hist(cross_per_lake, bins=15)
# plt.show()
# sys.exit()
n_prof_str = [str(n) for n in n_profiles]


# n_prof_str = [str(n) for n in n_profiles]
# for site_ct, site_id in enumerate(lakes.values):
#   print(site_ct)
#   spdata = pgml_data[pgml_data['site_id'] == site_id]
#   tdata = transfer_data[transfer_data['site_id'] == site_id]
#   median_select = tdata['best_pred'].values[0]
#   worst_select = tdata['highest'].values[0]
#   best_select = tdata['lowest'].values[0]
#   med = np.array([spdata[str(n_prof)+' obs median'].values[0] for n_prof in n_prof_str])
#   std = np.array([spdata[str(n_prof)+' obs std'].values[0] for n_prof in n_prof_str])
#   med[med < 0.1] = np.nan
#   std[med < 0.1] = np.nan
#   plt.errorbar(n_prof_str, med, yerr=std, color='black',xerr=None, fmt='', ecolor=None, elinewidth=None, \
#           capsize=None, lolims=False, uplims=False, xlolims=False, \
#           xuplims=False, errorevery=1, capthick=None, data=None, label="PGML Using Target Observations")
#   # plt.fill_between(n_prof_str, worst_selected, y2=best_selected, alpha=0.2,color='cyan',where=None, interpolate=False, step=None, data=None)
#   # plt.axhline(y=worst_select, color='red', linestyle='-', label="Worst Actual Source out of top 10 predicted")
#   # plt.axhline(y=best_select, color='green', linestyle='-', label="Best Actual Source out of top 10 predicted")
#   ens_rmse = plot_data[plot_data['target_id'] == site_id]['ens_rmse'].values[0]
#   glm_rmse = plot_data[plot_data['target_id'] == site_id]['glm_rmse'].values[0]
#   pbmtl_rmse = plot_data[plot_data['target_id'] == site_id]['glm_t_rmse'].values[0]
#   plt.axhline(y=median_selected, color='green', linestyle='-', label="PG-MTL RMSE")
#   plt.axhline(y=ens_rmse, color='red', linestyle='-', label="PG-MTL9 RMSE")
#   plt.axhline(y=glm_rmse, color='blue', linestyle='-', label="PB0 RMSE")
#   plt.axhline(y=pbmtl_rmse, color='purple', linestyle='-', label="PB-MTL RMSE")
#   # pdb.set_trace()
#   # plt.fill_between(0, median_selected-0.739, y2=median_selected+0.739, color='cyan',alpha=0.3)
#   plt.xlabel("Number of Profiles Used")
#   plt.ylabel("Median RMSE across 305 test lakes")
#   plt.title("Target PGML versus Transferred Source")
#   plt.legend()
#   plt.savefig("./plots/"+site_id)
#   plt.clf()
green = '#b2df8a'

red = '#33a02c'

plt.errorbar(n_prof_str, med_per_n_prof,yerr=None, color='black',xerr=None, fmt='', ecolor=None, elinewidth=None, \
        capsize=None, lolims=False, uplims=False, xlolims=False, \
        xuplims=False, errorevery=1, capthick=None, data=None, label="PGDL Using Target Observations")
# plt.errorbar(n_prof_str, med_per_n_prof, yerr=mad_per_n_prof, color='black',xerr=None, fmt='', ecolor=None, elinewidth=None, \
#         capsize=None, lolims=False, uplims=False, xlolims=False, \
#         xuplims=False, errorevery=1, capthick=None, data=None, label="PGDL Using Target Observations")
print(med_per_n_prof)
print(mad_per_n_prof)
# plt.fill_between(n_prof_str, worst_selected, y2=best_selected, alpha=0.2,color='cyan',where=None, interpolate=False, step=None, data=None)
# plt.axhline(y=worst_selected, color='red', linestyle='-', label="Worst Actual Source out of top 10 predicted")
# plt.axhline(y=best_selected, color='green', linestyle='-', label="Best Actual Source out of top 10 predicted")
# lq = np.array([2.15446337, 2.0719993 , 1.88850053, 1.77046298, 1.72798523,
#        1.66575086, 1.64758348, 1.60221577, 1.61337011, 1.52187778,
#        1.48335056, 1.47726903])
# uq2 = np.array([2.8079669 , 2.6550325 , 2.43580534, 2.24498336, 2.19234924,
#        2.11810599, 2.07217698, 2.0548102 , 2.01588733, 1.92513068,
#        1.84940096, 1.74906777])

# uq = np.array([3.0497413 , 2.89187458, 2.64384781, 2.53085613, 2.44983977,
#        2.3854153 , 2.38047313, 2.3667349 , 2.3354856 , 2.29234397,
#        2.29750235, 2.25618542])
# lq2 = np.array([2.39649122, 2.28938051, 2.1281057 , 2.01781588, 1.95835807,
#        1.9284937 , 1.93670152, 1.88953983, 1.8735805 , 1.77074885,
#        1.71492323, 1.63093888])
# print((uq-lq).mean())
# print((uq2-lq2).mean())

median_single = np.median(plot_data['single_rmse'].values)
lq_single = np.quantile(plot_data['single_rmse'].values,.25)
uq_single = np.quantile(plot_data['single_rmse'].values,.75)
median_ens = np.median(plot_data['ens_rmse'].values)
lq_ens = np.quantile(plot_data['ens_rmse'].values,.25)
uq_ens = np.quantile(plot_data['ens_rmse'].values,.75)

plt.axhline(y=median_single, color=green, path_effects=[pe.Stroke(linewidth=4, foreground='black'), pe.Normal()], linestyle='-', linewidth=3,label="Top 1 Source Median RMSE (PGDL-MTL1)")
plt.axhline(y=median_ens, color=red, path_effects=[pe.Stroke(linewidth=4, foreground='black'), pe.Normal()], linestyle='-', linewidth=3, label="Top 9 Source Ensemble Median RMSE (PGDL-MTL9)")

plt.fill_between([0,11], lq_single, y2=uq_single, color=green,alpha=.7,hatch='\\')
plt.fill_between([0,11], lq_ens, y2=uq_ens, color=red,alpha=.5,hatch='/')
plt.fill_between(range(len(n_profiles)),y1=lq_per_n_prof,y2=uq_per_n_prof,color='black',alpha=0.3)
plt.ylim(0,3)
plt.xlim(0,11)
plt.xlabel("Number of Profiles Used")
plt.ylabel("Median RMSE across 305 test lakes")
plt.title("Target PGDL versus PGDL-MTL and PGDL-MTL9")
plt.legend(bbox_to_anchor=(0.8,0.25))
plt.show()

