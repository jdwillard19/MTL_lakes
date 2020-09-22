import numpy as np
from collections import deque
from pandas import DataFrame
from pandas import concat
import math
import sys
from sklearn.metrics import mean_squared_error
from datetime import datetime, timedelta
import pdb

#-> X_trn[seq_num,:,:] = feat_mat[d,start_index:end_index,:]

#this file contains useful functions for transforming data and also useful calculations, sorted alphabetically

# def rmse(predictions, targets):
#     if np.isnan(np.array(((predictions - targets)) ** 2)).all():
#         return np.nan
#     else:
#         return np.sqrt(((predictions - targets) ** 2).mean())
def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())    
def buildTrainAndTestForRNN(mat,seq_length,n_features,train_split=0.8, output_phys_loss_data=False):
    #this function splits the data into train and test, and optionally includes the matching
    #depths and dates
    feat_mat = mat['Depth_Time_Series_Features']
    Y_mat = mat['Depth_Time_Series_Labels']
    n_depths = len(set(mat['Depth_int'].flat))
    depth_values = np.array(np.sort(list(set(mat['Depth'].flat))))
    udates = np.sort(mat['udates'])

    seq_per_depth = math.floor((np.shape(mat['Modeled_temp_int'])[0] / seq_length) / n_depths)
    train_seq_per_depth = int(train_split*seq_per_depth)
    test_seq_per_depth = seq_per_depth - train_seq_per_depth
    n_train_seq = train_seq_per_depth * n_depths
    n_test_seq = test_seq_per_depth * n_depths
    n_seq = seq_per_depth * n_depths

    #build train and test sets
    X_trn = np.empty(shape=(n_train_seq, seq_length, n_features))
    X_tst = np.empty(shape=(n_test_seq, seq_length, n_features))
    Y_trn = np.empty(shape=(n_train_seq, seq_length, 1))
    Y_tst = np.empty(shape=(n_test_seq, seq_length, 1))

    #keep track of depth and date for physical loss calculation later
    tst_depths = np.empty(shape=(n_test_seq, seq_length,1))
    tst_dates = np.empty(shape=(n_test_seq, seq_length,1))
    trn_depths = np.empty(shape=(n_train_seq, seq_length,1))
    trn_dates = np.empty(shape=(n_train_seq, seq_length,1))
    udates = np.sort(mat['udates'])
    n_seq = seq_per_depth * n_depths

    for d in range(0,n_depths):
        #print("depth ",d)
        #for each depth fill in train and test data based on parameters
        for s in range(0,train_seq_per_depth):
            start_index = s*seq_length #e.g. s=0 -> 0, s=1->204
            end_index = (s+1)*seq_length #e.g. s=0 ->204, s=1 ->408
            seq_num = d*train_seq_per_depth + s #index to 
            X_trn[seq_num,:,:] = feat_mat[d,start_index:end_index,:]
            Y_trn[seq_num,:,0] = Y_mat[d,start_index:end_index,0]
            trn_depths[seq_num,:,0] = depth_values[d]
            trn_dates[seq_num,:,0] = udates[start_index:end_index,0]
        for s in range(train_seq_per_depth,seq_per_depth):
            start_index = s*seq_length
            end_index = (s+1)*seq_length
            seq_num = d*test_seq_per_depth + (s-train_seq_per_depth)
            X_tst[seq_num,:,:] = feat_mat[d,start_index:end_index,:]
            Y_tst[seq_num,:,0] = Y_mat[d,start_index:end_index,0]
            tst_depths[seq_num,:,0] = depth_values[d]
            tst_dates[seq_num,:,0] = udates[start_index:end_index,0]
    if output_phys_loss_data:
        #if we want to output the aligned depths and dates to the test set
        return (X_trn, X_tst, Y_trn, Y_tst, trn_depths, trn_dates, tst_depths, tst_dates)
    else:
        return (X_trn, X_tst, Y_trn, Y_tst)


def buildTrainAndTestForRNN_sw(mat,seq_length,n_features,train_split=0.8,
                               output_phys_loss_data=False, win_size = 1):
    #this function splits the data into train and test, and optionally includes the matching
    #depths and dates, this uses sliding window 
    feat_mat = mat['Depth_Time_Series_Features']
    Y_mat = mat['Depth_Time_Series_Labels']
    n_depths = len(set(mat['Depth_int'].flat))
    depth_values = np.array(np.sort(list(set(mat['Depth'].flat))))
    udates = np.sort(mat['udates'])

    seq_per_depth = math.floor((np.shape(mat['Modeled_temp_int'])[0] / seq_length) / n_depths)
    print(seq_per_depth)
    train_seq_per_depth = math.ceil(train_split*seq_per_depth) - 1
    test_seq_per_depth = seq_per_depth - math.ceil(train_split*seq_per_depth)
    print(train_seq_per_depth, ':', test_seq_per_depth)
    win_per_seq = math.floor(seq_length / win_size) - 1 #windows per sequence (only training)
    n_train_seq = train_seq_per_depth * n_depths * win_per_seq
    n_test_seq = test_seq_per_depth * n_depths
    n_seq = seq_per_depth * n_depths

    #build train and test sets
    X_trn = np.empty(shape=(n_train_seq, seq_length, n_features))
    X_tst = np.empty(shape=(n_test_seq, seq_length, n_features))
    Y_trn = np.empty(shape=(n_train_seq, seq_length, 1))
    Y_tst = np.empty(shape=(n_test_seq, seq_length, 1))

    #keep track of depth and date for physical loss calculation later
    tst_depths = np.empty(shape=(n_test_seq, seq_length,1))
    tst_dates = np.empty(shape=(n_test_seq, seq_length,1))
    trn_depths = np.empty(shape=(n_train_seq, seq_length,1))
    trn_dates = np.empty(shape=(n_train_seq, seq_length,1))
    udates = np.sort(mat['udates'])
    n_seq = seq_per_depth * n_depths

    tr_seq_ind = 0 #seq index for training data
    ts_seq_ind = 0
    for d in range(0,n_depths):
        #print("depth ",d)
        #for each depth fill in train and test data based on parameters
        for s in range(0,train_seq_per_depth): #minus 1 to account for sliding window
            start_index = s*seq_length #e.g. s=0 -> 0, s=1->204
            end_index = (s+1)*seq_length #e.g. s=0 ->204, s=1 ->408
            assert end_index < udates.size
            #seq_num = d*train_seq_per_depth + s #index to 
            for w in range(0, win_per_seq):
                win_start_ind = start_index+w*win_size
                win_end_ind = win_start_ind + seq_length
                X_trn[tr_seq_ind,:,:] = feat_mat[d,win_start_ind:win_end_ind,:]
                Y_trn[tr_seq_ind,:,0] = Y_mat[d,win_start_ind:win_end_ind,0]
                trn_depths[tr_seq_ind,:,0] = depth_values[d]
                trn_dates[tr_seq_ind,:,0] = udates[win_start_ind:win_end_ind,0]
                tr_seq_ind += 1
        if n_test_seq != 0:
            for s in range(train_seq_per_depth,seq_per_depth-1):
                start_index = s*seq_length
                end_index = (s+1)*seq_length
                X_tst[ts_seq_ind,:,:] = feat_mat[d,start_index:end_index,:]
                Y_tst[ts_seq_ind,:,0] = Y_mat[d,start_index:end_index,0]
                tst_depths[ts_seq_ind,:,0] = depth_values[d]
                tst_dates[ts_seq_ind,:,0] = udates[start_index:end_index,0]
                ts_seq_ind += 1
    assert tr_seq_ind == n_train_seq
    assert ts_seq_ind == n_test_seq

    if output_phys_loss_data:
        #if we want to output the aligned depths and dates to the test set
        return (X_trn, X_tst, Y_trn, Y_tst, trn_depths, trn_dates, tst_depths, tst_dates)
    else:
        return (X_trn, X_tst, Y_trn, Y_tst)

def buildTrainAndTestForRNN_sw_obs(mat_o, mat, seq_length,n_features,train_split=0.8,
                               output_phys_loss_data=False, win_size = 1):
    #Description: this function splits the data into train and test, and optionally includes the matching
    #depths and dates, this uses sliding window 
    #
    #Parameters: @mat_o: observed data matlab file (assumes you ran RNN_preprocess_obs)
    #            @mat: sampled data matlab file

    feat_mat = mat['Depth_Time_Series_Features']
    Y_mat = mat_o['Depth_Time_Series_Labels']
    n_depths = len(set(mat['Depth_int'].flat))
    depth_values = np.array(np.sort(list(set(mat['Depth'].flat))))
    udates = np.sort(mat['udates'])

    seq_per_depth = math.floor((np.shape(mat['Modeled_temp_int'])[0] / seq_length) / n_depths)
    print(seq_per_depth)
    train_seq_per_depth = math.ceil(train_split*seq_per_depth) - 1
    test_seq_per_depth = seq_per_depth - math.ceil(train_split*seq_per_depth)
    print(train_seq_per_depth, ':', test_seq_per_depth)
    win_per_seq = math.floor(seq_length / win_size) - 1 #windows per sequence (only training)
    n_train_seq = train_seq_per_depth * n_depths * win_per_seq
    n_test_seq = test_seq_per_depth * n_depths
    n_seq = seq_per_depth * n_depths

    #build train and test sets
    X_trn = np.empty(shape=(n_train_seq, seq_length, n_features))
    X_tst = np.empty(shape=(n_test_seq, seq_length, n_features))
    Y_trn = np.empty(shape=(n_train_seq, seq_length, 1))
    Y_tst = np.empty(shape=(n_test_seq, seq_length, 1))

    #keep track of depth and date for physical loss calculation later
    tst_depths = np.empty(shape=(n_test_seq, seq_length,1))
    tst_dates = np.empty(shape=(n_test_seq, seq_length,1))
    trn_depths = np.empty(shape=(n_train_seq, seq_length,1))
    trn_dates = np.empty(shape=(n_train_seq, seq_length,1))
    udates = np.sort(mat['udates'])
    n_seq = seq_per_depth * n_depths

    tr_seq_ind = 0 #seq index for training data
    ts_seq_ind = 0
    for d in range(0,n_depths):
        #print("depth ",d)
        #for each depth fill in train and test data based on parameters
        for s in range(0,train_seq_per_depth): #minus 1 to account for sliding window
            start_index = s*seq_length #e.g. s=0 -> 0, s=1->204
            end_index = (s+1)*seq_length #e.g. s=0 ->204, s=1 ->408
            assert end_index < udates.size
            #seq_num = d*train_seq_per_depth + s #index to 
            for w in range(0, win_per_seq):
                win_start_ind = start_index+w*win_size
                win_end_ind = win_start_ind + seq_length
                X_trn[tr_seq_ind,:,:] = feat_mat[d,win_start_ind:win_end_ind,:]
                Y_trn[tr_seq_ind,:,0] = Y_mat[d,win_start_ind:win_end_ind,0]
                trn_depths[tr_seq_ind,:,0] = depth_values[d]
                trn_dates[tr_seq_ind,:,0] = udates[win_start_ind:win_end_ind,0]
                tr_seq_ind += 1
        if n_test_seq != 0:
            for s in range(train_seq_per_depth,seq_per_depth-1):
                start_index = s*seq_length
                end_index = (s+1)*seq_length
                X_tst[ts_seq_ind,:,:] = feat_mat[d,start_index:end_index,:]
                Y_tst[ts_seq_ind,:,0] = Y_mat[d,start_index:end_index,0]
                tst_depths[ts_seq_ind,:,0] = depth_values[d]
                tst_dates[ts_seq_ind,:,0] = udates[start_index:end_index,0]
                ts_seq_ind += 1
    assert tr_seq_ind == n_train_seq
    assert ts_seq_ind == n_test_seq

    if output_phys_loss_data:
        #if we want to output the aligned depths and dates to the test set
        return (X_trn, X_tst, Y_trn, Y_tst, trn_depths, trn_dates, tst_depths, tst_dates)
    else:
        return (X_trn, X_tst, Y_trn, Y_tst)


def buildTrainAndTestForRNN_byDepth_sw(mat,seq_length,n_features,train_split=0.8,
                               output_phys_loss_data=False, win_size = 1, skip_seq=0):
    #this function splits the data into train and test, and optionally includes the matching
    #depths and dates, this uses sliding window and organizes such that every 51 samples contains
    #every depth
    feat_mat = mat['Depth_Time_Series_Features']
    Y_mat = mat['Depth_Time_Series_Labels']
    n_depths = len(set(mat['Depth_int'].flat))
    depth_values = np.array(np.sort(list(set(mat['Depth'].flat))))
    udates = np.sort(mat['udates'])

    seq_per_depth = math.floor((np.shape(mat['Modeled_temp_int'])[0] / seq_length) / n_depths)
    #print(seq_per_depth)
    train_seq_per_depth = math.ceil(train_split*(seq_per_depth)) - 1
    test_seq_per_depth = seq_per_depth - math.ceil(train_split*(seq_per_depth)) - skip_seq
    print("train seq ", train_seq_per_depth)
    print(" test seq: ", test_seq_per_depth)
    print(train_seq_per_depth, ':', test_seq_per_depth)
    win_per_seq = math.floor(seq_length / win_size) - 1 #windows per sequence (only training)
    n_train_seq = train_seq_per_depth * n_depths * win_per_seq
    n_test_seq = test_seq_per_depth * n_depths
    n_seq = seq_per_depth * n_depths

    #build train and test sets
    X_trn = np.empty(shape=(n_train_seq, seq_length, n_features))
    X_tst = np.empty(shape=(n_test_seq, seq_length, n_features))
    Y_trn = np.empty(shape=(n_train_seq, seq_length, 1))
    Y_tst = np.empty(shape=(n_test_seq, seq_length, 1))

    #keep track of depth and date for physical loss calculation later
    tst_depths = np.empty(shape=(n_test_seq, seq_length,1))
    tst_dates = np.empty(shape=(n_test_seq, seq_length,1))
    trn_depths = np.empty(shape=(n_train_seq, seq_length,1))
    trn_dates = np.empty(shape=(n_train_seq, seq_length,1))
    udates = np.sort(mat['udates'])
    n_seq = seq_per_depth * n_depths

    tr_seq_ind = 0 #seq index for training data
    ts_seq_ind = 0

    for s in range(skip_seq,train_seq_per_depth+skip_seq):
        start_index = s*seq_length
        end_index = (s+1)*seq_length
        assert end_index < udates.size
        for w in range(0, win_per_seq):
            win_start_ind = start_index + w*win_size
            win_end_ind = win_start_ind + seq_length
            for d in range(0,n_depths):
                X_trn[tr_seq_ind, :, :] = feat_mat[d,win_start_ind:win_end_ind,:]
                Y_trn[tr_seq_ind,:,0] = Y_mat[d,win_start_ind:win_end_ind,0]
                trn_depths[tr_seq_ind,:,0] = depth_values[d]
                trn_dates[tr_seq_ind,:,0] = udates[win_start_ind:win_end_ind,0]
                tr_seq_ind += 1
    if n_test_seq != 0:
        for s in range(train_seq_per_depth+skip_seq,seq_per_depth-1):
                # print(s)
                start_index = s*seq_length
                end_index = (s+1)*seq_length
                for d in range(0,n_depths):
                    X_tst[ts_seq_ind,:,:] = feat_mat[d,start_index:end_index,:]
                    Y_tst[ts_seq_ind,:,0] = Y_mat[d,start_index:end_index,0]
                    tst_depths[ts_seq_ind,:,0] = depth_values[d]
                    tst_dates[ts_seq_ind,:,0] = udates[start_index:end_index,0]
                    ts_seq_ind += 1
    assert tr_seq_ind == n_train_seq
    assert ts_seq_ind == n_test_seq


    if output_phys_loss_data:
        #if we want to output the aligned depths and dates to the test set
        assert Y_tst.shape[0] == tst_depths.shape[0] 
        assert Y_tst.shape[0] == tst_dates.shape[0]
        return (X_trn, X_tst, Y_trn, Y_tst, trn_depths, trn_dates, tst_depths, tst_dates)
    else:
        return (X_trn, X_tst, Y_trn, Y_tst)


def buildTrainAndTestForRNN_obs(mat,mat_s,seq_length,n_features,train_split=0.8, output_phys_loss_data=False):
    #this function splits the data into train and test, and optionally includes the matching
    #depths and dates
    #
    # seq_mat_feat = np.empty(shape=(n_depths, seq_length*seq_per_depth, n_features))
    # seq_mat_label = np.empty(shape=(n_depths, seq_length*seq_per_depth, 1))
    # #ignore remainder dates
    # ignore_dates = udates[-2:]
    # for i in range(0,n):
    #   if(i % 10000 == 0):
    #       print(i, " data processed")
    #   if mat['datenum'][i] in ignore_dates:
    #       #skip over ignored dates
    #       continue
    #   else:
    #       #get depth and datenum indices for matrix
    #       depth_ind = np.where(u_depth_values == depths[i])[0]
    #       datenum_ind = np.where(udates == datenums[i])[0]

    #       #place data
    #       seq_mat_feat[depth_ind,datenum_ind,:] = x[i,:]
    #       seq_mat_label[depth_ind,datenum_ind,0] = y[i]


    # #removes depths with not many labels
    # depth_observation_cutoff = 35
    # depth_rm_ind = []
    # ignored_depths = {}
    depth_values = np.array(np.sort(list(set(mat_s['Depth'].flat))))

    # for d in range(0,n_depths):
    #   if np.count_nonzero(seq_mat_label[d,:]) < depth_observation_cutoff:
    #       depth_rm_ind.append(d)
    #       ignored_depths.add(u_depth_values[d])


    # feat_mat = seq_mat_feat
    # Y_mat = seq_mat_label
    feat_mat = mat['Depth_Time_Series_Features']
    Y_mat = mat['Depth_Time_Series_Labels']
    n_depths = Y_mat.shape[0]
    n_dates = Y_mat.shape[1]

    seq_per_depth = math.floor((n_dates / seq_length))
    print(seq_per_depth)
    train_seq_per_depth = int(train_split*seq_per_depth)
    test_seq_per_depth = seq_per_depth - train_seq_per_depth
    n_train_seq = train_seq_per_depth * n_depths
    n_test_seq = test_seq_per_depth * n_depths
    n_seq = seq_per_depth * n_depths

    #build train and test sets
    X_trn = np.empty(shape=(n_train_seq, seq_length, n_features))
    X_tst = np.empty(shape=(n_test_seq, seq_length, n_features))
    Y_trn = np.empty(shape=(n_train_seq, seq_length, 1))
    Y_tst = np.empty(shape=(n_test_seq, seq_length, 1))

    #keep track of depth and date for physical loss calculation later
    tst_depths = np.empty(shape=(n_test_seq, seq_length,1))
    tst_dates = np.empty(shape=(n_test_seq, seq_length,1))
    trn_depths = np.empty(shape=(n_train_seq, seq_length,1))
    trn_dates = np.empty(shape=(n_train_seq, seq_length,1))
    udates = np.array(np.sort(list(set(mat_s['datenums'].flat))))
    print(train_seq_per_depth)
    assert udates.size == n_dates
    n_seq = seq_per_depth * n_depths

    for d in range(0,n_depths):
        #print("depth ",d)
        #for each depth fill in train and test data based on parameters
        for s in range(0,train_seq_per_depth):
            start_index = s*seq_length #e.g. s=0 -> 0, s=1->204
            end_index = (s+1)*seq_length #e.g. s=0 ->204, s=1 ->408
            seq_num = d*train_seq_per_depth + s #index to 
            X_trn[seq_num,:,:] = feat_mat[d,start_index:end_index,:]
            print(Y_mat[d,start_index:end_index,0])
            Y_trn[seq_num,:,0] = Y_mat[d,start_index:end_index,0]
            trn_depths[seq_num,:,0] = depth_values[d]
            print(trn_dates[seq_num,:,0].shape)
            print(udates.shape)
            trn_dates[seq_num,:,0] = udates[start_index:end_index,0]
        for s in range(train_seq_per_depth,seq_per_depth):
            start_index = s*seq_length
            end_index = (s+1)*seq_length
            seq_num = d*test_seq_per_depth + (s-train_seq_per_depth)
            X_tst[seq_num,:,:] = feat_mat[d,start_index:end_index,:]
            Y_tst[seq_num,:,0] = Y_mat[d,start_index:end_index,0]
            tst_depths[seq_num,:,0] = depth_values[d]
            tst_dates[seq_num,:,0] = udates[start_index:end_index]
    if output_phys_loss_data:
        #if we want to output the aligned depths and dates to the test set
        return (X_trn, X_tst, Y_trn, Y_tst, trn_depths, trn_dates, tst_depths, tst_dates)
    else:
        return (X_trn, X_tst, Y_trn, Y_tst)




def calculatePhysicalLossDensityDepth(yhat, depths, udates, dates):
        #
    #create reference matrix
    delta_it = createDensityDeltaMatrix(yhat, depths, udates, dates)
    #take reLU
    relu_delta_it = np.maximum(delta_it,0)
    #take average
    phys_loss = np.mean(relu_delta_it)
    return phys_loss



def calculateRMSE(y, yhat, start_index=0,end_index=-1):
    #calculates rmse given prediction(yhat) and original(y)

    if(end_index == -1):
        end_index = np.shape(y)[0]

    assert start_index >= 0 and start_index < end_index
    return math.sqrt(mean_squared_error(y[start_index:end_index],yhat[start_index:end_index]))


def createDensityMatrix(yhat,udates,u_depths, days, depths, verbose=False):
    #creates matrix with densities for each (depth,time) pair
    n = np.shape(yhat)[0]
    num_u_dates = np.shape(udates)[0]
    num_u_depths = np.shape(u_depths)[0]
    print("pred: ", n, ", dates: ",np.shape(depths)[0], ", days: ", np.shape(days)[0])
    assert n == np.shape(depths)[0] and n == np.shape(days)[0]
    d_mat = np.empty(shape=(num_u_depths,num_u_dates))
    for i in range(0,n):
        depth_ind = np.where(u_depths == depths[i])[0]
        day_ind = np.where(udates == days[i])[0]
        # print("depth_ind: ", depth_ind)
        # print("day ind: ", day_ind)
        if yhat.ndim == 2:
            d_mat[depth_ind, day_ind] = transformTempToDensity(yhat[i,0])
        elif yhat.ndim == 1:
            d_mat[depth_ind, day_ind] = transformTempToDensity(yhat[i])
        else:
            print("invalid yhat dimensions")

        if(i % 10000 == 0 and verbose): #verbosity
            print(i, " processed in creating density matrix")
    return d_mat

def createDensityDeltaMatrix(yhat, depths, udates, datenums, verbose=False):
    #creates matrix of density delta between (depth,time) pair and (depth+1,time) pair
    #parameters: yhat - input data
    #            depths - depth array that aligns with input data
    #            udates - array of unique dates
    #            
    udates = np.sort(udates)
    num_u_dates = np.shape(udates)[0]
    u_depth_values = np.array(np.sort(list(set(depths.flat))))
    num_u_depths = np.shape(u_depth_values)[0]
    density = createDensityMatrix(yhat, udates, u_depth_values, datenums, depths)
    dit_mat = np.empty(shape=(num_u_depths-1,num_u_dates))
    for d in range(0,num_u_depths-1):
        #print("processing dens diff at depth ", d)
        for t in range(0,num_u_dates):
            if(density[d,t] == 0 or density[d+1,t] == 0 or 
               np.isnan(density[d,t]) or np.isnan(density[d+1,t]) or 
               abs(density[d,t]) > 2000 or abs(density[d+1,t]) > 2000):
                dit_mat[d,t] = 0 #only relevent for missing data
            else:
                dit_mat[d,t] = density[d,t] - density[d+1,t]
    return dit_mat

def createTemperatureMatrix(yhat,udates,u_depths, days, depths, verbose=False):
    #creates matrix with densities for each (depth,time) pair
    #parameters
        #@yhat: temperature array
        #@udates: unique dates
        #@u_depths: unique depths
        #@days: dates aligned with yhat
        #@depths: depths aligned with yhat
    #creates matrix with densities for each (depth,time) pair
    udates = np.sort(udates)
    n = np.shape(yhat)[0]
    num_u_dates = np.shape(udates)[0]
    num_u_depths = np.shape(u_depths)[0]
    print("pred: ", n, ", dates: ",np.shape(depths)[0], ", days: ", np.shape(days)[0])
    assert n == np.shape(depths)[0] and n == np.shape(days)[0]
    d_mat = np.empty(shape=(num_u_depths,num_u_dates))
    for i in range(0,n):
        depth_ind = np.where(u_depths == depths[i])[0]
        day_ind = np.where(udates == days[i])[0]
        # print("depth_ind: ", depth_ind)
        # print("day ind: ", day_ind)
        if yhat.ndim == 2:
            d_mat[depth_ind, day_ind] = (yhat[i,0])
        elif yhat.ndim == 1:
            d_mat[depth_ind, day_ind] = (yhat[i])
        else:
            print("invalid yhat dimensions")

        if(i % 10000 == 0 and verbose): #verbosity
            print(i, " processed in creating temp matrix")
    return d_mat

def findThermoDepth( rhoVar,depths,Smin=0.1, seasonal=False ):
#Author: Jordan S Read 2009 ----
#updated 3 march 2011
#Adapted from matlab to python 
    #PARAMETERS:
        #rhoVar: density vector
        #depths: depth vector
        #Smin: cutoff value 
        #seasonal: return seasonal values or not

    dRhoPerc = 0.15 #min percentage max for unique thermocline step
    numDepths = depths.shape[0]
    drho_dz = np.empty((numDepths-1))
    drho_dz[:] = np.nan
    print("depth size, ", depths.shape)
    for i in range(0,numDepths-1):  
        drho_dz[i] = (rhoVar[i+1]-rhoVar[i])/(depths[i+1]-depths[i])

    if seasonal:
        #%look for two distinct maximum slopes, lower one assumed to be seasonal
        mDrhoZ = drho_dz.max()    #%find max slope
        thermoInd =  np.argmax(drho_dz)   
        thermoD = (depths[thermoInd]+depths[thermoInd+1])/2.0#%depth of max slope
        if thermoInd > 1 and thermoInd < numDepths-2: #%if within range, 
            Sdn = -(depths[thermoInd+1]-depths[thermoInd])/(drho_dz[thermoInd+1]-drho_dz[thermoInd]);
            Sup = (depths[thermoInd]-depths[thermoInd-1])/(drho_dz[thermoInd]-drho_dz[thermoInd-1]);
            upD  = depths[thermoInd];
            dnD  = depths[thermoInd+1];
            if not any([np.isinf(Sup), np.isinf(Sdn)]):
                #print("valid values")
                thermoD = dnD*(Sdn/(Sdn+Sup))+upD*(Sup/(Sdn+Sup))
            else:
                print("infinite gradient?")
        dRhoCut = np.array([dRhoPerc*mDrhoZ, Smin]).max()
        (pks,locs) = locPeaks(drho_dz,dRhoCut)
        if pks.size == 0:
            SthermoD = thermoD
            SthermoInd = thermoInd
        else:
            mDrhoZ = pks[pks.size-1] #is this right?
            SthermoInd = locs[pks.size-1]
            if SthermoInd > thermoInd+1:
                SthermoD = np.mean(np.array([depths[SthermoInd], depths[SthermoInd+1]]))
                if SthermoInd > 1 and SthermoInd < numDepths-1:
                    Sdn = -(depths[SthermoInd+1]-depths[SthermoInd])/(drho_dz[SthermoInd+1]-drho_dz[SthermoInd])
                    Sup = (depths[SthermoInd]-depths[SthermoInd-1])/(drho_dz[SthermoInd]-drho_dz[SthermoInd-1])
                    upD  = depths[SthermoInd]
                    dnD  = depths[SthermoInd+1]
                    if not any([np.isinf(Sup), np.isinf(Sdn)]):
                        SthermoD = dnD*(Sdn/(Sdn+Sup))+upD*(Sup/(Sdn+Sup))
            else:
                SthermoD = thermoD
                SthermoInd = thermoInd
        if SthermoD < thermoD:
            SthermoD = thermoD
            SthermoInd = thermoInd
        return  (thermoD,thermoInd,drho_dz,SthermoD,SthermoInd)
    else:
        mDrhoZ = drho_dz.max()    #%find max slope
        thermoInd = np.argmax(drho_dz)           
        thermoD = np.mean(np.array([depths[thermoInd], depths[thermoInd+1]]));   #epth of max slope
        if thermoInd > 1 and thermoInd < numDepths-1: #if within range, 
            Sdn = -(depths[thermoInd+1]-depths[thermoInd])/(drho_dz[thermoInd+1]-drho_dz[thermoInd])
            Sup = (depths[thermoInd]-depths[thermoInd-1])/(drho_dz[thermoInd]-drho_dz[thermoInd-1])
            upD  = depths[thermoInd]
            dnD  = depths[thermoInd+1]
            if not any([np.isinf(Sup), np.isinf(Sdn)]):
                thermoD = dnD*(Sdn/(Sdn+Sup))+upD*(Sup/(Sdn+Sup));
        return  (thermoD,thermoInd,drho_dz)

def getThermoclineVectors(u_depths, n_dates, data):
    # @data: depths x days temps
    n_depths = u_depths.size
    tvec = np.empty((n_dates))
    stvec = np.empty((n_dates.size))
    tvec[:] = np.nan
    stvec[:] = np.nan
    for d in range(0,n_dates):
        density_vector = transformTempToDensity(data[:,d])
        # print(density_vector)
        # if d==3:
        #   print(density_vector)
        depths = u_depths
        (thermoD,thermoInd,drho_dz,SthermoD,SthermoInd) = findThermoDepth(density_vector, depths, seasonal=True)
        # print('thermoD: ', thermoD)
        # print('thermoInd: ', thermoInd)
        tvec[d] = thermoInd
        stvec[d] = SthermoInd


    return (tvec, stvec)

def getThermoclineVector(u_depths, data):
    # @data: depths x days temps
    n_depths = u_depths.size
    tvec = np.empty((1))
    stvec = np.empty((1))
    tvec[:] = np.nan
    stvec[:] = np.nan

    density_vector = transformTempToDensity(data)
    # print(density_vector)
    # if d==3:
    #   print(density_vector)
    depths = u_depths
    (thermoD,thermoInd,drho_dz,_,_) = findThermoDepth(density_vector, depths, seasonal=True)
    # print('thermoD: ', thermoD)
    # print('thermoInd: ', thermoInd)
    tvec = thermoInd


    return tvec

def locPeaks(dataIn,dataMn):

    # %----Author: Jordan S Read 2011 ---- Adapted to Python by Jared Willard 2018

    # % this program attempts to mirror 'findpeaks.m' from the signal processing
    # % toolbox

    # % dataIn: vector of input data
    # % dataMn: threshold for peak height

    # % finds multiple peaks for dataIn
    # % peaks: peak values
    # % locs:  indices of peaks

    # % -- description --
    # % a peak is a peak if it represents a local maximum


    varL = dataIn.shape[0];
    locs = np.zeros((varL), dtype='bool')
    peaks = np.empty((varL))
    peaks[:] = np.nan
    for i in range(1,varL):
        #if i == 49:
            #print("HERE, ",dataIn[i-1:i+1].max())
        posPeak = dataIn[i-1:i+1].max()
        pkI = np.argmax(dataIn[i-1:i+1])
        if pkI == 1:
            #print("peak ", i, " set")
            peaks[i] = posPeak
            locs[i]  = True

    inds = np.arange(varL);
    locs = inds[locs];
    peaks= peaks[locs];

    #% remove all below threshold value
    useI = (peaks >= dataMn)
    peaks= peaks[useI]
    locs = locs[useI]
    return (peaks,locs)

def matlabDateToPythonDate(datenum):
    return datetime.fromordinal(int(matlab_datenum)) + timedelta(days=matlab_datenum%1) - timedelta(days = 366)

def transformTempToDensity(temp):
    # print(temp)
    #converts temperature to density
    #parameter:
        #@temp: single value or array of temperatures to be transformed
    if(isinstance(temp,list)):
        return [1000*(1-((t+288.9414)*(t - 3.9863)**2)/(508929.2*(t+68.12963))) for t in temp]
    else:
        return 1000*(1-((temp+288.9414)*(temp - 3.9863)**2)/(508929.2*(temp+68.12963)))






