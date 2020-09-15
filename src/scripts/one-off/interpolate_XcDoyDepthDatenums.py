import os.path
import numpy as np
import sys
sys.path.append('../../data')
sys.path.append('/home/invyz/workspace/Research/lake_monitoring/src/data')

from rw_data import readMatData, saveMatData
from data_operations import createTemperatureMatrix
######################################################
# this script was used to create the Xc_doy_int, Depths_int, datenums_int, modeled_temp_int
# which are interpolated version of xc_doy, Depths, datenums in 
# mendota_sampled.mat, respectively
###########################################################

mat = readMatData(os.path.abspath('../../../data/processed/mendota_sampled.mat'))
x = mat
udates = mat['udates']
d = {x: [] for x in udates[:,0]} #dict of unique days to depths, initialized to zeros
for n in range(0,np.shape(x['Depth'])[0]):
	d[x['datenums'][n][0]].append(x['Depth'][n][0])	#append depth value to day key

num_depths = 51
full_depths_set = {}


#get array of unique depths for reference, this script makes the assumption that 
#each day has one depth 
for k, v in d.items():   
    if len(set(v)) == num_depths:
        full_depths_set = set(v)
        break

#get standardized depth values - HARDCODED RN
depth_values = np.array(np.sort(list(set(mat['Depth'].flat))))

std_depth_values = mat['Xc_doy'][0:51,2]
surface_depth_standardized = min(std_depth_values) #normalized surface depth value

def getStandardizedFromOriginal(orig, std, x):
	orig = np.sort(orig)
	std = np.sort(std)
	return std[np.where(orig == x)[0]][0]

#get surface feature vector for each day
def getSurfaceFeatureVectorFromDay(numday,surface_depth_standardized):
	#observations x features matrix
	day_matrix = mat['Xc_doy'][np.where(mat['datenums']==numday),:][0,:] 
	#return element that corresponds to surface depth
	return day_matrix[day_matrix[:,2]==surface_depth_standardized]


#create new data items with these new depths using surface data and append them to new_Xc_doy
new_Xc_doy = mat['Xc_doy']
new_depths = mat['Depth']
new_datenums = mat['datenums']
new_modeled_temp = mat['Modeled_temp_tuned']
t_mat = createTemperatureMatrix(mat['Modeled_temp_tuned'],mat['udates'],depth_values, mat['datenums'], mat['Depth'], verbose=False)
added = 0
#for each unique day
for key, value in d.items():
	#print("day: ", key)
	day = key
	#if depths are missing
	if len(set(value)) < num_depths:
		# print("missing values found") #debug
		#acquire missing depth values
		missing_depths_set = full_depths_set.difference(set(value))

		#interpolate temp for missing depths
		#x values needed(depths)
		x = depth_values
		#x values we already have
		xp = np.setdiff1d(x,np.array(list(missing_depths_set)))
		#y values we already have
		fp = t_mat[np.where(np.in1d(x,xp))[0],np.where(udates == day)[0]]
		#do interpolation
		y = np.interp(x,xp,fp)
		#for each missing depth value
		for depth in missing_depths_set:
			# print("missing observation found on day ", key, " at depth ", depth)	#debug		
			#get standardized depth
			std_depth = getStandardizedFromOriginal(depth_values, std_depth_values, depth)
			
			#create new feature vector with new depth that was missing and the features
			#from the surface vector
			#add new depth the surface feature vec

			new_data_vector = getSurfaceFeatureVectorFromDay(key,surface_depth_standardized)
			new_data_vector[0,2] = std_depth

			#get new temperature from depth above


			#append to new_Xc_doy, new_depths, new_datenums
			new_Xc_doy = np.concatenate((new_Xc_doy, new_data_vector), axis=0)
			new_depths = np.vstack((new_depths, depth))
			new_datenums = np.vstack((new_datenums, key))
			new_temp = y[x == depth] 
			new_modeled_temp = np.vstack((new_modeled_temp, new_temp))

			added += 1
			#assert(np.shape(new_Xc_doy)[0] == np.shape(mat['Xc_doy'])[0] + added)

print("size of OG Xc_doy ", np.shape(mat['Xc_doy']))
print("size of new Xc_doy ", np.shape(new_Xc_doy))
print("elements added: ", added)
mat['Xc_doy_int'] = new_Xc_doy
mat['Depth_int'] = new_depths
mat['datenums_int'] = new_datenums
mat['Modeled_temp_tuned_int'] = new_modeled_temp

# interpolate values for modeled temperature
saveMatData(mat, os.path.abspath('../../../data/processed/mendota_sampled.mat'))