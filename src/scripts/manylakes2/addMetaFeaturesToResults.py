import pandas as pd
import numpy as np
import pdb



csv = pd.read_csv('./305_lake_results.csv')
csv = csv[['target_id0','target_id','pgmtl9_rmse','glm_rmse','rmse_pred_mean','rmse_pred_min']]
df.to_csv('./305_lake_results.csv')
