#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 10:28:42 2023

@author: eilers
"""

import pickle
import numpy as np

# -------------------------------------------------------------------------------
# load tiny data set with 31 objects
# -------------------------------------------------------------------------------

f = open('../RM_black_hole_masses/data_HST_1220_5000_2A.pickle', 'rb') 
data, data_ivar = pickle.load(f)
f.close()

# -------------------------------------------------------------------------------
# initialize parameters
# -------------------------------------------------------------------------------


for qq in range(data.shape[0]): 
 
    # remove one object for test set, rest as training set
    ind_test = np.zeros(data.shape[0], dtype = bool)
    ind_test[qq] = True
    ind_train = ~ind_test 

    # -------------------------------------------------------------------------------
    # re-scale all input data to be Gaussian with zero mean and unit variance
    # -------------------------------------------------------------------------------
    
    qs = np.nanpercentile(data[ind_train, :], (2.5, 50, 97.5), axis=0)
    pivots = qs[1]
    scales = (qs[2] - qs[0]) / 4.
    scales[scales == 0] = 1.
    data_scaled = (data - pivots) / scales
    data_ivar_scaled = data_ivar * scales**2 
    
    # -------------------------------------------------------------------------------
    # prepare rectangular training data
    # -------------------------------------------------------------------------------
    
    # spectra
    X = data_scaled[:, 7:] 
    X_var = 1 / data_ivar_scaled[:, 7:]
    
    # labels
    inds_label = np.zeros(data_scaled.shape[1], dtype = bool)
    inds_label[0] = True # black hole mass
    inds_label[1] = True # redshift
    inds_label[6] = True # Lbol
    Y = data_scaled[:, inds_label] 
    Y_var = (1 / data_ivar_scaled[:, inds_label])
    
    
    
# -------------------------------------------------------------------------------
# load data slightly larger data set with 1000 objects
# -------------------------------------------------------------------------------
     
hdu = fits.open('data_norm_sdss16_1000.fits')  
issues = hdu[4].data
wave = hdu[0].data  
X = hdu[1].data[issues == 0.]
# X_ivar = hdu[2].data[issues == 0.]
masks = hdu[3].data[issues == 0.]
Y = hdu[5].data[issues == 0.]

# set missing values to NaN
X[masks == 0.] = np.nan

# full data set will have 23085 quasars (or ~80000), only 1000 now
X = X[:1000, :]
Y = Y[:1000, :]

qs = np.nanpercentile(X, (2.5, 50, 97.5), axis=0)
pivots = qs[1]
scales = (qs[2] - qs[0]) / 4.
scales[scales == 0] = 1.
X = (X - pivots) / scales

qs = np.nanpercentile(Y, (2.5, 50, 97.5), axis=0)
pivots = qs[1]
scales = (qs[2] - qs[0]) / 4.
scales[scales == 0] = 1.
Y = (Y - pivots) / scales