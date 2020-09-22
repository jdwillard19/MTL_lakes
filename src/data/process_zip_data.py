import os
import shutil
import glob
import zipfile 
from zipfile import ZipFile 
import sys


#create folders if not existing
base_path = "../../data/raw/sb_mtl_data_release/"
path_extensions = ["ice_flags","meteo","predictions","cfg",\
				   "obs"]

for path_ext in path_extensions:
	path = base_path + path_ext
	if not os.path.exists(path):
		os.mkdir(path)



#unzip files into their respective directories
#obs
obs_zips = glob.glob(base_path + "*observations.zip")
for file_name in obs_zips:
	with ZipFile(file_name, 'r') as zip: 
	    # printing all the contents of the zip file 
	    zip.printdir() 
	  
	    # extracting all the files 
	    print('Extracting all the files now...') 
	    zip.extractall(base_path + "obs/") 
	    print('Done!') 

#predictions
pred_zips = glob.glob(base_path + "pb0_predictions*.zip")
for file_name in pred_zips:
	with ZipFile(file_name, 'r') as zip: 
	    # printing all the contents of the zip file 
	    zip.printdir() 
	  
	    # extracting all the files 
	    print('Extracting all the files now...') 
	    zip.extractall(path=base_path+"predictions") 
	    print('Done!') 

#config files (not a zip anymore, just move)
shutil.move(base_path+"pb0_config.json", base_path+"cfg/")

#meteo
input_zips = glob.glob(base_path+"inputs*.zip")
for file_name in input_zips:
	with ZipFile(file_name, 'r') as zip: 
	    # printing all the contents of the zip file 
	    zip.printdir() 
	  
	    # extracting all the files 
	    print('Extracting all the files now...') 
	    zip.extractall(path=base_path+"meteo/") 
	    print('Done!') 


#ice flags
iflag_zips = glob.glob(base_path+"ice*.zip")
for file_name in iflag_zips:
	with ZipFile(file_name, 'r') as zip: 
	    # printing all the contents of the zip file 
	    zip.printdir() 
	  
	    # extracting all the files 
	    print('Extracting all the files now...') 
	    zip.extractall(path=base_path+"ice_flags/") 
	    print('Done!') 

