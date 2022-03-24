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
from sklearn.decomposition import PCA
from functions_s_gplvm import PCAInitial, lnL_Z_new, mean_var, lnL_h, Chi2_Matrix, NN, predictY, lnL_h_band, lnL

# -------------------------------------------------------------------------------
# plotting settings
# -------------------------------------------------------------------------------

colors = ["windows blue", "amber", "greyish", "faded green", "dusty purple", "pale red", "orange", "red", "blue", "lime green", "deep blue", "carolina blue"]
colors = sns.xkcd_palette(colors)

fsize = 24
matplotlib.rcParams['ytick.labelsize'] = fsize
matplotlib.rcParams['xtick.labelsize'] = fsize
matplotlib.rc('text', usetex=True)

colors_cont = ["black", "grey", "light grey"] 
colors_cont = sns.xkcd_palette(colors_cont)

# -------------------------------------------------------------------------------
# parameters
# -------------------------------------------------------------------------------

seed = 42
np.random.seed(seed)

labels = list(['$\log_{10}\,(M_{\\bullet}/M_\odot)$']) #, '$z$'])
plot_limits = {}
plot_limits['$\log_{10}\,(M_{\\bullet}/M_\odot)$'] = (5, 10)
plot_limits['$z$'] = (0, 2)
plot_limits['$\log_{10}\,L_{\\rm bol}$'] = (40, 50)


# -------------------------------------------------------------------------------
# load data
# -------------------------------------------------------------------------------

# first entries: black hole masses, z, SNR, BAL, survey, normalization mean, Lbol
# remaining entries: spectra
f = open('data_HST_SDSS_bin2_5000.pickle', 'rb')
data, data_ivar = pickle.load(f)
f.close()


# -------------------------------------------------------------------------------
# cuts
# -------------------------------------------------------------------------------


# cuts_z = data[:, 1] < 5
cuts_SNR = data[:, 2] > 0
cuts_BAL = data[:, 3] == 0
#cuts_survey = data[:, 4] == 5 # np.logical_or(data[:, 4] == 1, data[:, 4] == 5) # data[:, 4] < 6 # 
#cuts_Lbol = data[:, 6] < 47
# cuts_wave_norm = data[:, 5] == 2500

data = data[cuts_BAL * cuts_SNR] # * cuts_survey * cuts_Lbol]
data_ivar = data_ivar[cuts_BAL * cuts_SNR] # * cuts_survey * cuts_Lbol]

'''
1: G17_SDSSRM_Hbeta
2: G17_SDSSRM_Halpha
3: G19_SDSSRM_CIV
4: H20_SDSSRM_MgII
5: HST (Hbeta)
6: X-Shooter
'''

# ps: pivor & scaled input data
name = 'ps_s{}'.format(seed) 

cross = True

# -------------------------------------------------------------------------------
# re-scale all input data to be Gaussian with zero mean and unit variance
# -------------------------------------------------------------------------------

qs = np.nanpercentile(data, (2.5, 50, 97.5), axis=0)
pivots = qs[1]
scales = (qs[2] - qs[0]) / 4.
data_scaled = (data - pivots) / scales
data_ivar_scaled = data_ivar * scales**2 

# -------------------------------------------------------------------------------
# prepare rectangular training data
# -------------------------------------------------------------------------------

X = data_scaled[:, 6:]
X_var = 1 / data_ivar_scaled[:, 6:]
inds_label = np.zeros(data_scaled.shape[1], dtype = bool)
inds_label[0] = True
Y = data_scaled[:, inds_label] 
Y_var = (1 / data_ivar_scaled[:, inds_label]) 

# -------------------------------------------------------------------------------
# initialize parameters
# -------------------------------------------------------------------------------

if cross:
    all_Y_new = np.zeros_like(Y)
    all_Y_new_var = np.zeros_like(Y)
    QQ = X.shape[0]
    name_addition = '_cross'
else:
    QQ = 1
    ind_test = np.zeros(X.shape[0], dtype = bool)
    ind_test = np.random.choice(X.shape[0], size = 10, replace = False)
    ind_train = np.ones(X.shape[0], dtype=bool)
    ind_train[ind_test] = False
    #ind_test[-N_hiz:] = True
    name_addition = '_HST_SDSS'

for qq in range(QQ): 
 
    if cross: 
        ind_test = np.zeros(X.shape[0], dtype = bool)
        ind_test[qq] = True
        ind_train = np.ones(X.shape[0], dtype=bool)
        ind_train[ind_test] = False
    
    print('training set: {} objects'.format(np.sum(ind_train)))
    print('testing set: {} objects'.format(np.sum(ind_test)))
    
    X_input = X[ind_train]
    X_var_input = X_var[ind_train]
    Y_input = Y[ind_train]
    Y_var_input = Y_var[ind_train]
    
    assert np.min(np.sum(np.isfinite(X_var)*np.isfinite(X), axis = 0)) > 0
    
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
    
    theta_rbf = 1. 
    gamma_rbf = 1. 
    theta_band, gamma_band = 1., 1.
    hyper_params = np.ones(4) #np.ones(D+L)
    
    # -------------------------------------------------------------------------------
    # initialize parameters
    # -------------------------------------------------------------------------------
    
    
    print('hyper parameters: theta_rbf={}, theta_band={}, gamma_rbf={}, gamma_band={}'.format(theta_rbf, theta_band, gamma_rbf, gamma_band))
    
    assert np.any(X_var != 0)
    
    Z_initial = PCAInitial(X_input, Q)
    Z = np.reshape(Z_initial, (N*Q,))
    
    name = 'N{}_D{}_L{}_Q{}_tband{}_gband{}_3000_bin2_{}_pivotscaled_seed31_newerrors_MCMCpred_bounds_optall_hyper1_HST_modified2'.format(N, D, L, Q, theta_band, gamma_band, name_addition)

    # -------------------------------------------------------------------------------
    # optimization in one
    # -------------------------------------------------------------------------------
    
    x0 = np.hstack((Z, hyper_params))
    bounds_min = np.ones_like(x0) * (-1e1)
    bounds_min[(N*Q):] = 1e-3 # avoid negative hyperparameters
    bnds = [(bounds_min[b], None) for b in range(0, len(bounds_min))]
    
    t1 = time.time()
    
    res = op.minimize(lnL, x0 = x0, args = (X_input, Y_input, Z_initial, X_var_input, Y_var_input, X_mask, Y_mask), 
                      method = 'L-BFGS-B', jac = True, bounds = bnds,      
                      options={'gtol':1e-12, 'ftol':1e-12, 'maxiter':10})
    
    print(res.success)
    print(res.message)
    pars_opt = res.x
    
    t2 = time.time()
    print('optimization in {} s.'.format(t2-t1))
    
    Z_final = np.reshape(pars_opt[:(N*Q)], (N, Q))
    theta_rbf = pars_opt[(N*Q):(N*Q)+1]
    gamma_rbf = pars_opt[(N*Q)+1:(N*Q)+2]
    theta_band = pars_opt[(N*Q)+2:(N*Q)+3]
    gamma_band = pars_opt[(N*Q)+3:(N*Q)+4]
    
    print('new hyper parameters: ', theta_rbf, gamma_rbf, theta_band, gamma_band)
    print('new latent parameters: ', Z_final)
    
    # -------------------------------------------------------------------------------
    # optimization 
    # -------------------------------------------------------------------------------
    
    # bnds = [(1e-5, None) for b in range(0, len(hyper_params))]
    
    # t1 = time.time()
    
    # max_iter = 3
    # for t in range(max_iter):
        
    #     t1a = time.time()
    #     # optimize hyperparameters
    #     print("optimizing hyper parameters")
    #     res = op.minimize(lnL_h, x0 = hyper_params, args = (X_input, Y_input, Z_initial, X_var_input, Y_var_input, X_mask, Y_mask), 
    #                       method = 'L-BFGS-B', jac = True, bounds = bnds,
    #                       options={'gtol':1e-12, 'ftol':1e-12})
        
    #     #print("optimizing hyper band parameters")
    #     #res = op.minimize(lnL_h_band, x0 = hyper_params, args = (theta_rbf, gamma_rbf, X_input, Y_input, Z_initial, X_var_input, Y_var_input, X_mask, Y_mask), method = 'L-BFGS-B', jac = True, 
    #     #                  options={'gtol':1e-12, 'ftol':1e-12})
        
    #     # update hyperparameters
    #     hyper_params = res.x
    #     print(res.message)
    #     print('success: {}'.format(res.success))
    #     #print('new hyper parameters: {}'.format(res.x))
    #     t2a = time.time()
    #     print('first optimization in {} s.'.format(t2a-t1a))
        
    #     # optimize Z
    #     print("optimizing latent parameters")
    #     res = op.minimize(lnL_Z_new, x0 = Z, args = (X_input, Y_input, hyper_params, theta_band, gamma_band, Z_initial, X_var_input, Y_var_input, X_mask, Y_mask), method = 'L-BFGS-B', jac = True, 
    #                       options={'gtol':1e-6, 'ftol':1e-9, 'maxiter': 1e4})
                 
    #     # update Z
    #     Z = res.x
    #     print(res.message)
    #     print('success: {}'.format(res.success))
    #     t3a = time.time()
    #     print('second optimization in {} s.'.format(t3a-t2a))
        
    # t2 = time.time()
    # print('optimization in {} s.'.format(t2-t1))
    
    # Z_final = np.reshape(Z, (N, Q))
    
    # -------------------------------------------------------------------------------
    # inferring new labels of training objects  
    # -------------------------------------------------------------------------------
    
    # Y_new_scaled = np.zeros_like(Y_input)
    # Y_var_new_scaled = np.zeros_like(Y_input)
    # for j in range(N):   
    #     mean, var, foo = mean_var(Z_final, Z_final[j, :], Y_input, Y_var_input, gamma_rbf, gamma_band)
    #     Y_new_scaled[j, :] = mean
    #     Y_var_new_scaled[j, :] = var
    # Y_new = Y_new_scaled * scales[inds_label] + pivots[inds_label]
    # Y_var_new = Y_var_new_scaled * scales[inds_label]**2 

    # -------------------------------------------------------------------------------
    # corner plots
    # -------------------------------------------------------------------------------
    
    all_pearsonr = np.zeros(Q)
    for n in range(Q):
        all_pearsonr[n] = pearsonr(Z_final[:, n], data[ind_train, 0])[0]
        
    sort_r = np.argsort(np.abs(all_pearsonr))[::-1]
    
    NNN = min(Q, 5)
    fig = plt.figure(figsize = (12, 15))
    widths = np.ones(NNN)
    widths[-1] = .1
    gs = gridspec.GridSpec(NNN, NNN, width_ratios = widths)
    gs.update(wspace = 0., hspace = 0.)
    
    for cols in range(NNN):   
        for rows in range(cols):
            print(sort_r[cols], sort_r[rows])
            ax = plt.subplot(gs[NNN-1-rows, NNN-1-cols])
            sc = ax.scatter(Z_final[:, sort_r[cols]], Z_final[:, sort_r[rows]], c = data[ind_train, 0])
            if rows == 0:
                ax.set_xlabel(r'$Z_{{\,{}}}$'.format(sort_r[cols]), fontsize = 22)
            if cols == NNN-1:
                ax.set_ylabel(r'$Z_{{\rm{}}}$'.format(sort_r[rows]), fontsize = 22)
            if NNN-1-cols > 0:
                ax.tick_params(labelleft = False)
            if rows > 0:
                ax.tick_params(labelbottom = False)
    
    cbar = plt.colorbar(sc, cax = plt.subplot(gs[NNN-1, NNN-1]))
    cbar.set_label(r'$\log_{10}(M_\bullet/M_\odot)$', fontsize = 22)
            
    plt.savefig('plots/new/corners/corner_{0}_{1}.pdf'.format(qq, name), bbox_inches = 'tight') 
    plt.close()
        
    # -------------------------------------------------------------------------------
    # prediction for new test object
    # -------------------------------------------------------------------------------
        
    print('prediction for test objects...')
    
    Y_new_test = np.zeros((np.sum(ind_test), L))
    Y_new_var_test = np.zeros((np.sum(ind_test), L))
    Z_new_test = np.zeros((np.sum(ind_test), Q))
    N_new = np.sum(ind_test) #1 
    
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
        
        #good_stars_new = X_mask_new[i, :]
        Z_new_n, success_z, Y_new_n, Y_new_var_n, Y_new16, Y_new84, samples = predictY(X_new[n, :], X_var_new[n, :], X_input, X_var_input, Y_input, Y_var_input, Z_final, hyper_params, theta_band, gamma_band, y0, z0, qq, name, X_mask, Y_mask, X_mask_new[n])
        Y_new_test[n, :] = Y_new_n * scales[inds_label] + pivots[inds_label]
        Y_new_var_test[n, :] = Y_new_var_n * scales[inds_label]**2
        Z_new_test[n, :] = Z_new_n
        print('new Y!!! Y = ', Y_new_n * scales[inds_label] + pivots[inds_label], 'pm ', np.sqrt(Y_new_var_n * scales[inds_label]**2), '-- original: ', data[qq, 0])
    
    # Y_orig = data[ind_test, :L]
    # Y_var_orig = 1. / data_ivar[ind_test, :L]
    
    # for i, l in enumerate(labels):
    
    #     scatter = np.round(np.std(Y_orig[:, i] - Y_new_test[:, i]), 4)
    #     bias = np.round(np.mean(Y_orig[:, i] - Y_new_test[:, i]), 4)    
    #     chi2_label = np.round(np.sum((Y_orig[:, i] - Y_new_test[:, i])**2 / Y_var_orig[:, i]), 4)
        
    #     xx = [-10, 10]
    #     plt.figure(figsize=(6, 6))
    #     plt.scatter(Y_orig[:, i], Y_new_test[:, i], color=colors[-2], label=' bias = {0} \n scatter = {1} \n $\chi^2$ = {2}'.format(bias, scatter, chi2_label), marker = 'o')
    #     plt.plot(xx, xx, color=colors[2], linestyle='--')
    #     plt.xlabel(r'reference labels {}'.format(l), size=fsize)
    #     plt.ylabel(r'inferred values {}'.format(l), size=fsize)
    #     plt.tick_params(axis=u'both', direction='in', which='both')
    #     plt.xlim(plot_limits[l])
    #     plt.ylim(plot_limits[l])
    #     plt.tight_layout()
    #     plt.legend(loc=2, fontsize=14, frameon=True)
    #     #plt.title('#stars: {0}, #pixels: {1}, $\\theta_{{band}} = {4}$, $\\theta_{{rbf}} = {5}$, $\gamma_{{band}} = {6}$, $\gamma_{{rbf}} = {7}$'.format(len(ind_validation), D, L, Q, theta_band, theta_rbf_name, gamma_band, gamma_rbf_name), fontsize=11)
    #     plt.savefig('plots/1to1_test_{0}_{1}.png'.format(i, name))


    if cross:
        all_Y_new[qq] = Y_new_n * scales[inds_label] + pivots[inds_label]
        all_Y_new_var[qq] = Y_new_var_n * scales[inds_label]**2
        print(all_Y_new[qq], data[qq, 0])

if cross:
    Y_orig = data[:, inds_label]
    Y_var_orig = 1. / data_ivar[:, inds_label]
else:
    all_Y_new = Y_new_test        
    Y_orig = data[ind_test, :][:, inds_label]
    # add 0.4**2 only for high-z quasars
    Y_var_orig = 1. / data_ivar[ind_test, :][:, inds_label] #+ 0.4**2
    
for i, l in enumerate(labels):

    scatter = np.round(np.std(Y_orig[:, i] - all_Y_new[:, i]), 4)
    bias = np.round(np.mean(Y_orig[:, i] - all_Y_new[:, i]), 4)    
    chi2_label = np.round(np.sum((Y_orig[:, i] - all_Y_new[:, i])**2 / (Y_var_orig[:, i])) / N, 4) # 0.3**2 from unkown geometry factor
    
    xx = [-1000, 1000]
    plt.figure(figsize=(6, 6))
    #plt.scatter(Y_orig[:, i], all_Y_new[:, i], color=colors[-2], label=' bias = {0} \n scatter = {1} \n $\chi^2$ = {2}'.format(bias, scatter, chi2_label), marker = 'o')
    plt.errorbar(all_Y_new[:, i], Y_orig[:, i], yerr = np.sqrt((Y_var_orig[:, i])), xerr = np.sqrt((all_Y_new_var[:, i])), fmt = 'o', color=colors[0], label=' bias = {0} \n scatter = {1} \n reduced $\chi^2$ = {2}'.format(bias, scatter, chi2_label))
    plt.plot(xx, xx, color=colors[2], linestyle='--')
    plt.ylabel(r'measured labels {}'.format(l), size=fsize)
    plt.xlabel(r'inferred values {}'.format(l), size=fsize)
    plt.tick_params(axis=u'both', direction='in', which='both')
    plt.xlim(plot_limits[l])
    plt.ylim(plot_limits[l])
    plt.tight_layout()
    plt.legend(loc=4, fontsize=18, frameon=True)
    #plt.title('$\#$QSOs: {0}, #pixels: {1}, $\\theta_{{band}} = {4}$, $\\theta_{{rbf}} = {5}$, $\gamma_{{band}} = {6}$, $\gamma_{{rbf}} = {7}$'.format(len(ind_validation), D, L, Q, theta_band, theta_rbf_name, gamma_band, gamma_rbf_name), fontsize=11)
    plt.title(r'$N={},\,D={},\,L={},\,Q={}$'.format(N, D, L, Q), fontsize = 18)
    plt.savefig('plots/new/1to1_test_{0}_{1}.pdf'.format(i, name))
    plt.close()
    
# -------------------------------------------------------------------------------'''
# prediction for given black hole mass?
# -------------------------------------------------------------------------------

# Y_new_orig_all = np.array([6, 6.5, 7, 7.5, 8, 8.5, 9])
# X_new = np.zeros((len(Y_new_orig_all), D))

# for i, Y_new_orig in enumerate(Y_new_orig_all):
    
#     Y_new = (Y_new_orig - pivots[inds_label]) / scales[inds_label]
    
#     #Y_train = data_scaled[ind_train, :][:, inds_label]
#     #x0_ind = min(Y_train[:, 0], key=lambda x:abs(x-Y_new))
#     #x0 = X_input[np.where(Y_train[:, 0] == x0_ind)[0][0]]    
#     #z0 = Z_final[np.where(Y_train[:, 0] == x0_ind)[0][0], :]
    
#     x0 = X_input[-1, :]  # can't have nan's in there...
#     z0 = Z_final[-1, :]
            
#     Z_new, X_new_scaled = predictX(Y_new, X_input, X_var_input, Y_input, Y_var_input, Z_final, hyper_params, x0, z0, X_mask, Y_mask)
#     X_new[i, :] = X_new_scaled * scales[6:] + pivots[6:]


# min_waves, max_waves = 1220, 3000.1 
# wave_grid = np.arange(min_waves, max_waves, 2)

# fig, ax = plt.subplots(1, 1, figsize = (10, 6))
# for i, q in enumerate(list([2, 4, 6])):
#     ax.plot(wave_grid[2:], X_new[q, 1:], color = colors[i], label = r'$\log_{{10}} (M_\bullet/M_\odot) = {}$'.format(Y_new_orig_all[q]))
# ax.legend(frameon = True, fontsize = 18)
# ax.set_xlabel(r'$\lambda_{\rm rest}~\rm [{\AA}]$', fontsize = fsize)
# ax.set_ylabel(r'flux', fontsize = fsize)
# ax.set_xlim(min_waves, max_waves)
# plt.savefig('plots/new/predictX_{}.pdf'.format(name), bbox_inches = 'tight')

# log_Lbols = X_new[:, 0]


# -------------------------------------------------------------------------------'''
# plot of outliers
# -------------------------------------------------------------------------------

'''min_waves, max_waves = 1220, 3000.1 
wave_grid = np.arange(min_waves, max_waves, 2)
waves = wave_grid[:-1]+ 0.5* np.diff(wave_grid)

# for QQ = 1
outliers = np.array([26, 27, 29])
non_outliers = np.ones(data.shape[0], dtype = bool)
non_outliers[outliers] = False
composite = np.nanmean(data[non_outliers, 6:], axis = 0)

fig, ax = plt.subplots(1, 1, figsize = (12, 5))
ax.plot(waves, data[26, 6:], lw = 1, color = colors[0], label = 'outliers in latent space')
ax.plot(waves, data[27, 6:], lw = 1, color = colors[-1])
ax.plot(waves, data[29, 6:], lw = 1, color = colors[-2])
ax.plot(waves, composite, lw = 2, color = colors[5], label = 'composite spectrum')
ax.set_xlim(waves[0], waves[-1])
ax.set_xlabel(r'rest-frame wavelength [{\AA}]', fontsize = fsize)
ax.set_ylabel(r'flux', fontsize = fsize)
ax.legend(fontsize = fsize)
plt.savefig('plots/new/outliers_{}.pdf'.format(name), bbox_inches = 'tight')

# -------------------------------------------------------------------------------'''


    # -------------------------------------------------------------------------------
    # visualisation
    # -------------------------------------------------------------------------------
    
    # # latent space color coded by labels
    # for i, l in enumerate(labels):
    #     q = 0
    #     plt.figure(figsize=(9, 6))
    #     plt.tick_params(axis=u'both', direction='in', which='both')
    #     cm = plt.cm.get_cmap('viridis')
    #     sc = plt.scatter(Z_final[:, q], Z_final[:, q+1], c = Y_input[:, i], marker = 'o', cmap = cm)
    #     cbar = plt.colorbar(sc)
    #     cbar.set_label(r'{}'.format(labels[i]), rotation=270, size=fsize, labelpad = 10)
    #     plt.xlabel(r'latent dimension {}'.format(q), fontsize = fsize)
    #     plt.ylabel(r'latent dimension {}'.format(q+1), fontsize = fsize)
    #     plt.title(r'$\#$QSOs: {0}, $\#$pixels: {1}, $\#$labels: {2}, $\#$latent dim.: {3}, $\theta_{{\rm band}} = {4}$, $\theta_{{\rm rbf}} = {5}$, $\gamma_{{\rm band}} = {6}$, $\gamma_{{\rm rbf}} = {7}$'.format(N, D, L, Q, np.round(theta_band, 3), np.round(theta_rbf, 3), np.round(gamma_band, 3), np.round(gamma_rbf, 3)), fontsize=12)
    #     plt.tight_layout()
    #     plt.savefig('plots/Latent2Label_{}_{}_{}.png'.format(i, qq, name))
    #     plt.close()     
        
    
    # for i, l in enumerate(labels):
    
    #     scatter = np.round(np.std(Y_input[:, i] * scales[inds_label][i] + pivots[inds_label][i] - Y_new[:, i]), 5)
    #     bias = np.round(np.mean(Y_input[:, i] * scales[inds_label][i] + pivots[inds_label][i] - Y_new[:, i]), 5) 
    #     chi2_label = np.round(np.sum((Y_input[:, i] * scales[inds_label][i] + pivots[inds_label][i] - Y_new[:, i])**2 / (Y_var_input[:, i] * scales[inds_label][i]**2)), 4)
        
        # xx = [-10, 10]
        # plt.figure(figsize=(6, 6))
        # plt.scatter(Y_input[:, i] * scales[inds_label][i] + pivots[inds_label][i], Y_new[:, i], color=colors[-2], label=' bias = {0} \n scatter = {1} \n $\chi^2$ = {2}'.format(bias, scatter, chi2_label), marker = 'o')
        # plt.plot(xx, xx, color=colors[2], linestyle='--')
        # plt.xlabel(r'reference labels {}'.format(l), size=fsize)
        # plt.ylabel(r'inferred values {}'.format(l), size=fsize)
        # plt.tick_params(axis=u'both', direction='in', which='both')
        # plt.xlim(plot_limits[l])
        # plt.ylim(plot_limits[l])
        # plt.tight_layout()
        # plt.legend(loc=2, fontsize=14, frameon=True)
        # plt.title(r'$\#$QSOs: {0}, $\#$pixels: {1}, $\#$labels: {2}, $\#$latent dim.: {3}, $\theta_{{\rm band}} = {4}$, $\theta_{{\rm rbf}} = {5}$, $\gamma_{{\rm band}} = {6}$, $\gamma_{{\rm rbf}} = {7}$'.format(N, D, L, Q, np.round(theta_band, 3), np.round(theta_rbf, 3), np.round(gamma_band, 3), np.round(gamma_rbf, 3)), fontsize=12)
        # plt.savefig('plots/1to1_{}_{}_{}.png'.format(i, qq, name))
        # plt.close()