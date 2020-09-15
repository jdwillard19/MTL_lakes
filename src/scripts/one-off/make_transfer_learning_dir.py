import pandas as pd
import os

#read lake metadata file to get all the lakenames
metadata = pd.read_csv("../../../metadata/lake_metadata.csv")
lakenames = [str(i) for i in metadata.iloc[:,0].values] # to loop through all lakes



for target_lake in lakenames:
	os.mkdir("../../../results/transfer_learning/target_"+target_lake)
	for source_lake in lakenames:
		if target_lake == source_lake:
			continue
		else:
			os.mkdir("../../../results/transfer_learning/target_"+target_lake+"/source_"+source_lake)