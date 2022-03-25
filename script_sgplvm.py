#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  9 11:30:38 2021

@author: eilers
"""

import sys
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.constants as const
from astropy.io import fits
from astropy.table import Table, Column
import scipy.interpolate as interpol
import astropy.cosmology as cosmo
from astropy.convolution import convolve, Box1DKernel
import matplotlib.patches as patches
from scipy.optimize import curve_fit
import matplotlib.gridspec as gridspec
from astropy.io import ascii
import scipy.optimize as op
from scipy.stats import pearsonr
import time
import pickle
import corner
import os
import multiprocessing
from sklearn.decomposition import PCA
from functions_s_gplvm import PCAInitial, mean_var, Chi2_Matrix, NN, predictY, lnL

# -------------------------------------------------------------------------------
# import loop parameters
# -------------------------------------------------------------------------------

# Q_start = int(sys.argv[-2])
# Q_end = int(sys.argv[-1])
# print(Q_start, Q_end)
Q_start, Q_end = 0, 1

# -------------------------------------------------------------------------------
# parameters
# -------------------------------------------------------------------------------

seed = 42
np.random.seed(seed)

labels = list(['$\log_{10}\,(M_{\\bullet}/M_\odot)$']) #, '$z$'])
plot_limits = {}
plot_limits['$\\log_{10}\\,(M_{\\bullet}/M_\\odot)$'] = (6, 10)
plot_limits['$z$'] = (0, 2)
plot_limits['$\log_{10}\,L_{\\rm bol}$'] = (40, 50)


# -------------------------------------------------------------------------------
# load data
# -------------------------------------------------------------------------------

# first entries: black hole masses, z, SNR, BAL, survey, normalization mean, Lbol
# remaining entries: spectra
f = open('../RM_black_hole_masses/data_HST_SDSS_1220_5000_2A.pickle', 'rb') 
#f = open('data_HST_SDSS_1220_5000_2A.pickle', 'rb') 
#f = open('data_HST_SDSS_flexwave2_1220_5000.pickle', 'rb')
data, data_ivar = pickle.load(f)
f.close()

# -------------------------------------------------------------------------------
# cuts
# -------------------------------------------------------------------------------


# # # # cuts_z = data[:, 1] < 5
# # cuts_SNR = data[:, 2] > 5
# # # #cuts_BAL = data[:, 3] == 0
# cuts_survey = data[:, 4] == 5 
# # # #cuts_Lbol = data[:, 6] < 47
# data = data[cuts_survey] 
# data_ivar = data_ivar[cuts_survey] 

# adding intrinsic scatter of 0.4 dex
# data_ivar[:, 0] = 1./ (1./data_ivar[:, 0] + 0.4**2)

'''
1: G17_SDSSRM_Hbeta
2: G17_SDSSRM_Halpha
3: G19_SDSSRM_CIV
4: H20_SDSSRM_MgII
5: HST (Hbeta)
6: X-Shooter
'''

cross = True

name1 = 'ps_s{}_HST_SDSS_1220_5000_2A_noSNRcut'.format(seed) 

# -------------------------------------------------------------------------------
# initialize parameters
# -------------------------------------------------------------------------------

if cross:
    all_Y_new = np.zeros((data.shape[0], 1))
    all_Y_new_var = np.zeros((data.shape[0], 1))
    name1 += '_cross'

for qq in range(Q_start, Q_end): 
 
    if cross: 
        ind_test = np.zeros(data.shape[0], dtype = bool)
        ind_test[qq] = True
        ind_train = ~ind_test
        
        # np.random.seed(42)
        # N_test = data.shape[0] - 100 #int(len(data)/2)
        # qq_test = np.random.choice(data.shape[0], size = N_test, replace = False)
        # ind_test = np.zeros(data.shape[0], dtype = bool)
        # ind_train = np.zeros(data.shape[0], dtype=bool)
        # if qq == 0:             
        #     ind_test[qq_test] = True
        #     ind_train = ~ind_test
        # if qq == 1:
        #     ind_train[qq_test] = True
        #     ind_test = ~ind_train

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
    
    X = data_scaled[:, 7:] # only spectra
    X_var = 1 / data_ivar_scaled[:, 7:]
    inds_label = np.zeros(data_scaled.shape[1], dtype = bool)
    inds_label[0] = True # black hole mass
    inds_label[6] = True # Lbol
    #inds_label[1] = True # redshift
    Y = data_scaled[:, inds_label] 
    Y_var = (1 / data_ivar_scaled[:, inds_label])
    
    print('training set: {} objects'.format(np.sum(ind_train)))
    print('testing set: {} objects'.format(np.sum(ind_test)))
    
    X_input = X[ind_train]
    X_var_input = X_var[ind_train]
    Y_input = Y[ind_train]
    Y_var_input = Y_var[ind_train]
    
    #assert np.min(np.sum(np.isfinite(X_var)*np.isfinite(X), axis = 0)) > 0
    
    N = X_input.shape[0]
    D = X_input.shape[1] 
    L = Y.shape[1]
    Q = 16 # chosen number of latent dimensions
    
    print('N={}, D={}, L={}, Q={}'.format(N, D, L, Q))
    
    # -------------------------------------------------------------------------------
    # missing data
    # -------------------------------------------------------------------------------
    
    X_mask = np.ones_like(X_input).astype(bool)
    X_mask[np.isnan(X_input)] = False
    
    Y_mask = np.ones_like(Y_input).astype(bool)
    Y_mask[np.isnan(Y_input)] = False
    
    # -------------------------------------------------------------------------------
    # hyper parameters
    # -------------------------------------------------------------------------------
    
    Ax = np.ones(1) # theta_rbf
    Ay = np.ones(L) #L # gamma_rbf
    Bx = np.ones(1) # theta_band
    By = np.ones(L) #L # gamma_band
        
    #hyper_params = np.hstack([Bx, By, Ax, Ay]) 
    hyper_params = np.hstack([Ax, Bx, Ay, By]) 
    
    # -------------------------------------------------------------------------------
    # initialize parameters
    # -------------------------------------------------------------------------------
    
    
    print('hyper parameters: Ax={}, Bx={}, Ay={}, By={}'.format(Ax, Bx, Ay, By))
    
    assert np.any(X_var != 0)
    
    Z_initial = PCAInitial(X_input, Q)
    Z = np.reshape(Z_initial, (N*Q,))
    
    name = name1 + '_N{}_D{}_L{}_Q{}_qq{}'.format(N, D, L, Q, qq)
    print(name)

    # -------------------------------------------------------------------------------
    # optimization in one go
    # -------------------------------------------------------------------------------
    
    x0 = np.hstack((Z, hyper_params))
    bounds_min = np.ones_like(x0) * (-100)
    bounds_min[(N*Q):] = 0.1 # avoid negative hyperparameters
    bnds = [(bounds_min[b], None) for b in range(0, len(bounds_min))]
    
    beta = D / L
    
    t1 = time.time()
    
    res = op.minimize(lnL, x0 = x0, args = (X_input, Y_input, Z_initial, X_var_input, Y_var_input, beta, X_mask, Y_mask), 
                      method = 'L-BFGS-B', jac = True, bounds = bnds, options={'gtol':1e-9, 'ftol':1e-9})
    
    print(res.success)
    print(res.message)
    pars_opt = res.x
    
    t2 = time.time()
    print('optimization in {} s.'.format(t2-t1))
    
    Z_final = np.reshape(pars_opt[:(N*Q)], (N, Q))
    
    Ax = pars_opt[-2*L-2]
    Bx = pars_opt[-2*L-1]
    Ay = pars_opt[-2*L:-L]
    By = pars_opt[-L:]
    
    hyper_params = np.array([Ax, Bx, Ay, By])
    
    print('new hyper parameters: ', Ax, Bx, Ay, By)
    #print('new latent parameters: ', Z_final)

    # -------------------------------------------------------------------------------
    # corner plots
    # -------------------------------------------------------------------------------
    
    all_pearsonr = np.zeros(Q)
    for n in range(Q):
        all_pearsonr[n] = pearsonr(Z_final[:, n], data[ind_train, 0])[0]
        
    sort_r = np.argsort(np.abs(all_pearsonr))[::-1]
        
    # -------------------------------------------------------------------------------
    # prediction for new test object
    # -------------------------------------------------------------------------------
        
    print('prediction for test objects...')
    
    Y_new_test = np.zeros((np.sum(ind_test), L))
    Y_new_var_test = np.zeros((np.sum(ind_test), L))
    Z_new_test = np.zeros((np.sum(ind_test), Q))
    N_new = np.sum(ind_test) 
    
    X_new = X[ind_test, :] 
    X_var_new = X_var[ind_test, :]
    
    X_mask_new = np.ones_like(X_new).astype(bool)
    X_mask_new[np.isnan(X_new)] = False
    
    chi2 = Chi2_Matrix(X_input, 1./X_var_input, X_new, 1./X_var_new)
    all_NN = np.zeros((np.sum(ind_test), L))
    
    all_chis = []
    
    for n in range(N_new):
        
        # starting_guess
        y0, index_n = NN(n, chi2, Y_input)
        z0 = Z_final[index_n, :]
        all_NN[n, :] = y0
        
        #Z_opt_n, success_z, samples = predictY(X_new[n, :], X_var_new[n, :], X_input, X_var_input, Y_input, Y_var_input, Z_final, hyper_params, y0, z0, qq, name, X_mask, Y_mask, X_mask_new[n, :])
        Z_opt_n, success_z = predictY(X_new[n, :], X_var_new[n, :], X_input, X_var_input, Y_input, Y_var_input, Z_final, hyper_params, y0, z0, name, X_mask, Y_mask, X_mask_new[n, :])
        
        #Z25, Z16, Z_new_n, Z84, Z975 = np.percentile(samples, (2.5, 16, 50, 84, 97.5), axis = 0)    
        print('optimized latents: ', Z_opt_n)
        #print('sampled latents: ', Z_new_n)
        
        # # corner and steps plot
        # fig, axes = plt.subplots(ndim, 1, sharex=True, figsize=(8, 20)) 
        # for l in range(0, ndim):
        #     axes[l].plot(sampler.chain[:, :, l].T, color="k", alpha=0.4)
        #     axes[l].tick_params(axis=u'both', direction='in', which='both')               
        # axes[-1].set_xlabel('step number') 
        # plt.savefig('/Users/eilers/Dropbox/projects/RM_black_hole_masses/plots/new/corners/steps_{}_{}.pdf'.format(i, name))  
        # plt.close()
                
        # fig = corner.corner(samples, quantiles = [0.16, 0.5, 0.84])
        # plt.savefig('/Users/eilers/Dropbox/projects/RM_black_hole_masses/plots/new/corners/samples_{}.pdf'.format(name)) 
        # plt.close()
        
        # no optimization required!
        for l in range(L):
            good_stars = Y_mask[:, l]
            Y_new_n, Y_new_var_n, k_Z_zj, factor = mean_var(Z_final, Z_opt_n, Y_input[good_stars, l], Y_var_input[good_stars, l], Ay[l], By[l])                
            Y_new_test[n, l] = Y_new_n * scales[inds_label][l] + pivots[inds_label][l]
            Y_new_var_test[n, l] = Y_new_var_n * scales[inds_label][l]**2
            print('new Y!!! Y = ', Y_new_test[n, l], 'pm ', np.sqrt(Y_new_var_test[n, l]), '-- original: ', data[ind_test][n][l])
        
        Z_new_test[n, :] = Z_opt_n
            
        # f = open('files/f_{}.pickle'.format(name), 'wb')
        # pickle.dump((Y_new_test, Y_new_var_test, Z_new_test, data[ind_test][:, 0], data_ivar[ind_test][:, 0]), f) #, sort_r, Z_final, Z_opt_n, hyper_params, res.success, success_z), f)
        # f.close()
        
    if cross:
        all_Y_new[qq, :] = Y_new_test 
        all_Y_new_var[qq, :] = Y_new_var_test
        print(all_Y_new[qq], data[qq, inds_label])   
    
    f = open('../RM_black_hole_masses/files/f_{}.pickle'.format(name), 'wb')
    # f = open('files/f_{}.pickle'.format(name), 'wb')
    pickle.dump((all_Y_new, all_Y_new_var, sort_r, Z_final, Z_opt_n, hyper_params, res.success, success_z), f)
    f.close()

    print(name)
    
# -------------------------------------------------------------------------------'''

# # test likelihood function! 
# eps = 1e-6
# l1 = lnL(x0, X_input, Y_input, Z_initial, X_var_input, Y_var_input, X_mask, Y_mask)
# for i in range(494, 501):
#     x0[i] += eps
#     l2 = lnL(x0, X_input, Y_input, Z_initial, X_var_input, Y_var_input, X_mask, Y_mask)
    
#     x0[i] -= 2*eps
#     l3 = lnL(x0, X_input, Y_input, Z_initial, X_var_input, Y_var_input, X_mask, Y_mask)
    
#     x0[i] += eps
#     print(l1[1][i] - ((l2[0] - l3[0])/(2*eps)))
        
# -------------------------------------------------------------------------------'''
