import numpy as np
import pandas as pd
import pdb
import os
import re


def getMeteoFileName(site_id):
    nml_data = []
    with open('../../../data/raw/usgs_zips/cfg/nhdhr_'+site_id+'.nml', 'r') as file:
        nml_data = file.read().replace('\n', '')
    return re.search('meteo_fl\s+=\s+\'(NLDAS.+].csv)\'', nml_data).group(1)
    
