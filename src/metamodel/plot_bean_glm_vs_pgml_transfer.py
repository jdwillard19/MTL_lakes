import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pdb


y_pg = pd.read_csv("compiledTestResults2.csv", header=None)
pdb.set_trace()
pg = [float(i) for i in y_pg[3].values[1:].tolist()]
glm = [float(i) for i in y_pg[8].values[1:].tolist()]
glm_u = [float(i) for i in y_pg[5].values[1:].tolist()]
pg.pop(48)
glm.pop(48)
glm_u.pop(48)
pg_x = ['PGML_transfer'] * len(pg)
glm_x = ['GLM_transfer'] * len(glm)
glm_u_x = ['GLM_uncal'] * len(glm_u)
x = glm_u_x  + glm_x + pg_x
y = (glm_u + glm + pg)

df = pd.DataFrame()
df2 = pd.DataFrame()
# x.reverse()
# x_sp.reverse()
# pdb.set_trace()
y.pop(48)
x.pop(48)
y.pop(138)
x.pop(138)
df['Method'] = x
df['RMSE (degrees C)'] = y  

# df2['Method'] = x_sp
# df2['RMSE (degrees C)'] = glm + y_full + y_200 + y_150 + y_100+ y_50   
ax = sns.violinplot(x="Method", y="RMSE (degrees C)", data=df, palette="muted", scale='width')
ax.set_autoscale_on(False)
# for i in range(90):

# 	plt.plot(['GLM', 'PGMTL'], [glm[i], pg[i]], color='black', linewidth=0.5)
	# plt.plot(['PGMTL_50','PGMTL_100', 'PGMTL_150', 'PGMTL_200', 'PGMTL_all', 'GLM'], [glm[i], y_full[i], y_200[i], y_150[i], y_100[i], y_50[i]], color='black', linewidth=1)
plt.show()

# data = pd.read_csv('bean_plot_data2.csv')
# data = pd.read_csv('./manylakes2/scatter_data.csv')
# df = pd.DataFrame()
# df2 = pd.DataFrame()
# pgmtl = data['pgmtl_rmse'].values
# pgmtl2 = data['pgmtl_top1'].values
# glm = data['glm_rmse'].values
# df['Method'] = x
# df['RMSE (degrees C)'] = np.concatenate((glm, y_full))  
# df2['Method'] = x_sp
# df2['RMSE (degrees C)'] = glm + y_full + y_200 + y_150 + y_100+ y_50   




sys.exit()



line_data = np.linspace(0,7,700)
pgmtl = pg
plt.scatter(glm, pgmtl, color='r')
# plt.scatter(glm, pgmtl2, color='b', label='Single Best Predicted')
plt.xlabel("GLM Error (deg C)")
plt.ylabel("PGMTL Error (deg C)")
plt.title("Scatterplot of GLM_transfer vs PGML_transfer Error")
plt.plot(line_data, line_data)
# plt.legend()
# for i in range(8):
# 	# plt.plot(['GLM', 'PGMTL'], [glm[i], y_full[i]], color='black', linewidth=1)
# 	plt.plot(['PGMTL_50','PGMTL_100', 'PGMTL_150', 'PGMTL_200', 'PGMTL_all', 'GLM'], [glm[i], y_full[i], y_200[i], y_150[i], y_100[i], y_50[i]], color='black', linewidth=1)
plt.show()