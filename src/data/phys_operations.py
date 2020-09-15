import math
import torch
import sys
import pandas as pd
import os

def calculate_air_density(air_temp, rh):
    #returns air density in kg / m^3
    #equation from page 13 GLM/GLEON paper(et al Hipsey)

    #Ratio of the molecular (or molar) weight of water to dry air
    mwrw2a = 18.016 / 28.966
    c_gas = 1.0e3 * 8.31436 / 28.966

    #atmospheric pressure
    p = 1013. #mb

    #water vapor pressure
    vapPressure = calculate_vapour_pressure_air(rh,air_temp)

    #water vapor mixing ratio (from GLM code glm_surface.c)
    r = mwrw2a * vapPressure/(p - vapPressure)
    # print( 0.348*(1+r)/(1+1.61*r)*(p/(air_temp+273.15)))
    # print("vs")
    # print(1.0/c_gas * (1 + r)/(1 + r/mwrw2a) * p/(air_temp + 273.15))
    # sys.exit()
    # return 0.348*(1+r)/(1+1.61*r)*(p/(air_temp+273.15))
    return (1.0/c_gas * (1 + r)/(1 + r/mwrw2a) * p/(air_temp + 273.15))*100# 
def calculate_heat_flux_sensible(surf_temp, air_temp, rel_hum, wind_speed):
    #equation 22 in GLM/GLEON paper(et al Hipsey)
    #GLM code ->  Q_sensibleheat = -CH * (rho_air * 1005.) * WindSp * (Lake[surfLayer].Temp - MetData.AirTemp);
    #calculate air density 
    rho_a = calculate_air_density(air_temp, rel_hum)

    #specific heat capacity of air in J/(kg*C)
    c_a = 1005.


    #bulk aerodynamic coefficient for sensible heat transfer
    c_H = 0.0013

    #wind speed at 10m
    U_10 = calculate_wind_speed_10m(wind_speed)
    # U_10 = wind_speed
    return -rho_a*c_a*c_H*U_10*(surf_temp - air_temp)
 
def calculate_heat_flux_latent(surf_temp, air_temp, rel_hum, wind_speed):
    #equation 23 in GLM/GLEON paper(et al Hipsey)
    #GLM code-> Q_latentheat = -CE * rho_air * Latent_Heat_Evap * (0.622/p_atm) * WindSp * (SatVap_surface - MetData.SatVapDef)
    # where,         SatVap_surface = saturated_vapour(Lake[surfLayer].Temp);
    #                rho_air = atm_density(p_atm*100.0,MetData.SatVapDef,MetData.AirTemp);
    #air density in kg/m^3
    rho_a = calculate_air_density(air_temp, rel_hum)

    #bulk aerodynamic coefficient for latent heat transfer
    c_E = 0.0013

    #latent heat of vaporization (J/kg)
    lambda_v = 2.453e6

    #wind speed at 10m height
    # U_10 = wind_speed
    U_10 = calculate_wind_speed_10m(wind_speed)
# 
    #ratio of molecular weight of water to that of dry air
    omega = 0.622

    #air pressure in mb
    p = 1013.

    e_s = calculate_vapour_pressure_saturated(surf_temp)
    e_a = calculate_vapour_pressure_air(rel_hum, air_temp)
    return -rho_a*c_E*lambda_v*U_10*(omega/p)*(e_s - e_a)

    
def calculate_vapour_pressure_air(rel_hum, temp):
    rh_scaling_factor = 1
    return rh_scaling_factor * (rel_hum / 100) * calculate_vapour_pressure_saturated(temp)

def calculate_vapour_pressure_saturated(temp):
    # returns in miilibars
    # print(torch.pow(10, (9.28603523 - (2332.37885/(temp+273.15)))))

    #Converted pow function to exp function workaround pytorch not having autograd implemented for pow
    exponent = torch.tensor((9.28603523 - (2332.37885/(temp+273.15)))*math.log(10))
    return torch.exp(exponent)

def calculate_wind_speed_10m(ws, ref_height=2.):
    #from GLM code glm_surface.c
    c_z0 = 0.001 #default roughness
    return ws*(math.log(10.0/c_z0)/math.log(ref_height/c_z0))



def getMaxDepth(target_id):
    my_path = os.path.abspath(os.path.dirname(__file__))

    raw_dir = '../../data/raw/figure3/' #unprocessed data directory
    raw_dir = os.path.join(my_path, raw_dir)

    m_glm_path = raw_dir+"nhd_"+target_id+"_temperatures_wEnergy.feather"
    m_glm = pd.read_feather(m_glm_path)
    m_glm_val = m_glm.values
    n_depths = m_glm_val.shape[1]-4 #minus date and ice flag and energies
    return 0.5*(n_depths-1)
