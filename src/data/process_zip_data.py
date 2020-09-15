import os
import shutil
import glob
import zipfile 
from zipfile import ZipFile 
import sys


#mtl data release
# os.mkdir("../../data/raw/sb_mtl_data_release/ice_flags")
# os.mkdir("../../data/raw/sb_mtl_data_release/inputs")
# os.mkdir("../../data/raw/sb_mtl_data_release/pb0_predictions")
# os.mkdir("../../data/raw/sb_mtl_data_release/predictions")



# # #pgdl data release
# os.mkdir("../../data/raw/sb_pgdl_data_release/predictions")
# os.mkdir("../../data/raw/sb_pgdl_data_release/cfg")
# os.mkdir("../../data/raw/sb_pgdl_data_release/meteo")
# os.mkdir("../../data/raw/sb_pgdl_data_release/ice_flags")
# os.mkdir("../../data/raw/sb_pgdl_data_release/obs")
# # # os.mkdir("../../metadata")




  

# #obs
# #predictions
obs_zips = glob.glob("../../data/raw/sb_pgdl_data_release/*observations.zip")
for file_name in obs_zips:
	with ZipFile(file_name, 'r') as zip: 
	    # printing all the contents of the zip file 
	    zip.printdir() 
	  
	    # extracting all the files 
	    print('Extracting all the files now...') 
	    zip.extractall(path="../../data/raw/sb_pgdl_data_release/obs/") 
	    print('Done!') 

sys.exit()
#predictions
pred_zips = glob.glob("../../data/raw/sb_pgdl_data_release/predictions*.zip")
for file_name in pred_zips:
	with ZipFile(file_name, 'r') as zip: 
	    # printing all the contents of the zip file 
	    zip.printdir() 
	  
	    # extracting all the files 
	    print('Extracting all the files now...') 
	    zip.extractall(path="../../data/raw/sb_pgdl_data_release/predictions") 
	    print('Done!') 

#config files
nml_zips = glob.glob("../../data/raw/sb_pgdl_data_release/*_nml_files.zip")
shutil.move("../../data/raw/sb_pgdl_data_release/pb0_config.json", "../../data/raw/sb_pgdl_data_release/cfg/")
shutil.move("../../data/raw/sb_pgdl_data_release/pball_config.json", "../../data/raw/sb_pgdl_data_release/cfg/")
for file_name in nml_zips:
	with ZipFile(file_name, 'r') as zip: 
	    # printing all the contents of the zip file 
	    zip.printdir() 
	  
	    # extracting all the files 
	    print('Extracting all the files now...') 
	    zip.extractall(path="../../data/raw/sb_pgdl_data_release/cfg/") 
	    print('Done!') 



#meteo
input_zips = glob.glob("../../data/raw/sb_pgdl_data_release/inputs*.zip")
for file_name in input_zips:
	with ZipFile(file_name, 'r') as zip: 
	    # printing all the contents of the zip file 
	    zip.printdir() 
	  
	    # extracting all the files 
	    print('Extracting all the files now...') 
	    zip.extractall(path="../../data/raw/sb_pgdl_data_release/meteo") 
	    print('Done!') 


#ice
iflag_zips = glob.glob("../../data/raw/sb_pgdl_data_release/ice*.zip")
for file_name in iflag_zips:
	with ZipFile(file_name, 'r') as zip: 
	    # printing all the contents of the zip file 
	    zip.printdir() 
	  
	    # extracting all the files 
	    print('Extracting all the files now...') 
	    zip.extractall(path="../../data/raw/sb_pgdl_data_release/ice_flags/") 
	    print('Done!') 

