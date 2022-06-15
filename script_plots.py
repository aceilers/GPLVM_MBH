#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 12 08:31:56 2021

@author: eilers
"""

from os.path import exists
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import corner
import cmasher as cmr
from matplotlib import cm
import matplotlib.gridspec as gridspec
from astropy.io import fits
import itertools
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from functions_s_gplvm import mean_var, predictX


# -------------------------------------------------------------------------------
# plotting settings
# -------------------------------------------------------------------------------

fsize = 22
matplotlib.rcParams['ytick.labelsize'] = fsize
matplotlib.rcParams['xtick.labelsize'] = fsize
matplotlib.rc('text', usetex=True)

colors = ["windows blue", "amber", "greyish", "faded green", "dusty purple", "pale red", "dark red"]
colors = sns.xkcd_palette(colors)

np.random.seed(42)

# -------------------------------------------------------------------------------
# xxx
# ------------------------------------------------------------------------------- 

labels = list(['$\log_{10}\,(M_{\\bullet}/M_\odot)$', '$\log_{10}\,L_{\\rm bol}$', '$z$', '$\\lambda_{\\rm Edd}$'])
plot_limits = {}
plot_limits['$\log_{10}\,(M_{\\bullet}/M_\odot)$'] = (5.5, 9.5)
plot_limits['$z$'] = (0, 2)
plot_limits['$\\tau$'] = (-100, 200)
plot_limits['$\log_{10}\,L_{\\rm bol}$'] = (40, 50)
plot_limits['$\\lambda_{\\rm Edd}$'] = (0, 1.5)


f = open('../RM_black_hole_masses/data_HST_1220_5000_2A.pickle', 'rb')
data, data_ivar = pickle.load(f)
f.close()

# # SDSS data only
# data = data[:-31, :]
# data_ivar = data_ivar[:-31, :]

# # HST data only
# data = data[-31:, :]
# data_ivar = data_ivar[-31:, :]

# data_pred = data[-6:, :]

# # split last 6 spectra in two halves!
# min_waves, max_waves = 1220, 5000.1
# wave_grid = np.arange(min_waves, max_waves, 2)[1:-1]
# data_new = np.ones((6, data.shape[1]))
# data_ivar_new = np.zeros((6, data.shape[1]))
# split_wave = 1500
# cut_split_low = np.ones((data.shape[1]), dtype = bool)
# cut_split_low[7:][wave_grid > split_wave] = False
# cut_split_hi = np.ones((data.shape[1]), dtype = bool)
# cut_split_hi[7:][wave_grid < split_wave] = False
# for i, j in enumerate(range(25, 31)):
#     print(i, j)
#     data_new[i, cut_split_hi] = data[j, cut_split_hi]
#     data_ivar_new[i, cut_split_hi] = data_ivar[j, cut_split_hi]

#     data[j, ~cut_split_low] = np.nan
#     data_new[i, ~cut_split_hi] = np.nan
#     data_ivar[j, ~cut_split_low] = np.nan
#     data_ivar_new[i, ~cut_split_hi] = np.nan
    
# data_all = np.vstack([data[:25, :], data_new])
# data_ivar_all = np.vstack([data_ivar[:25, :], data_ivar_new])

# data = data_all
# data_ivar = data_ivar_all  


# data = data[-12:, :]
# data_ivar = data_ivar[-12:, :]

N = data.shape[0]
D = data.shape[1]
L = 3
Q = 16

lam = False
lag = False
flexwave = False
predict = False

if lag:
    name = 'f_ps_s42_HST_SDSS_1220_5000_2A_band1fixed_lag_noSNRcut_beta10_cross_N75_D1889_L{}_Q{}'.format(L, Q)
elif predict:
    name = 'f_ps_s42_HST_predictSDSS_1220_5000_2A_band1fixed_noSNRcut_beta10_new_2kernels_N31_D1889_L{}_Q{}'.format(L, Q)
else:
    name = 'f_ps_s42_HST_1220_5000_2A_band1fixed_noSNRcut_beta10_new_2kernels_lambda_cross_N{}_D1889_L{}_Q{}'.format(N-1, L, Q)

all_Y_new_all = np.zeros((N, 1))
all_Y_new_var_all = np.zeros((N, 1))
suc = np.zeros((N))

for qq in range(N):
    if exists('files/{}_qq{}.pickle'.format(name, qq)):
        f = open('files/{}_qq{}.pickle'.format(name, qq), 'rb')
        all_Y_new_i, all_Y_new_var_i, sort_r, Z_final, Z_opt_n, hyper_params, success_train, success_test = pickle.load(f)
        f.close()
        if success_train:
            suc[qq] = 1
        print(success_train, success_test)
        all_Y_new_all[qq, :] = all_Y_new_i[qq]
        all_Y_new_var_all[qq, :] = all_Y_new_var_i[qq]
    else:
        print('file missing for N={}!'.format(qq))

# all_Y_new_all = all_Y_new_all[-12:, :] 
# all_Y_new_var_all = all_Y_new_var_all[-12:, :]   

for ll in range(1):
    
    # black holes or lags
    if ll == 0:        
        if lam:
            l = labels[3]
            Y_orig = data[:, 5]
            Y_var_orig = 1./data_ivar[:, 5]
        else:
            l = labels[0]
            Y_orig = data[:, 0]
            Y_var_orig = 1./data_ivar[:, 0]
        all_Y_new = all_Y_new_all[:, ll]
        all_Y_new_var = all_Y_new_var_all[:, ll]
    
    # # Lbol
    # if ll == 1:
    #     l = labels[1]
    #     Y_orig = data[:, 6]
    #     Y_var_orig = 1./data_ivar[:, 6]
    #     all_Y_new = all_Y_new_all[:, ll]
    #     all_Y_new_var = all_Y_new_var_all[:, ll]
        
    # # z
    # if ll == 2:
    #     l = labels[2]
    #     Y_orig = data[:, 1]
    #     Y_var_orig = 1./data_ivar[:, 1]
    #     all_Y_new = all_Y_new_all[:, ll]
    #     all_Y_new_var = all_Y_new_var_all[:, ll]

    masked = all_Y_new > 0
    all_Y_new = all_Y_new[masked] 
    all_Y_new_var = all_Y_new_var[masked]
    masked_orig = data[:, 2] > 0 #5
    data = data[masked_orig]
    Y_orig = Y_orig[masked_orig][masked]  
    Y_var_orig = Y_var_orig[masked_orig][masked]    
    
    scatter = np.round(np.std(Y_orig - all_Y_new), 3)
    bias = np.round(np.mean(Y_orig - all_Y_new), 3)    
    chi2_label = np.round(np.sum((Y_orig - all_Y_new)**2 / (all_Y_new_var + Y_var_orig)) / N, 4) # 0.3**2 from unkown geometry factor
    
    xx = [-1000, 1000]
    cdict = {1.: colors[0], 2.: colors[1], 5.: colors[5]}
    leg = 0
    cmap = 'viridis'
    fig, ax = plt.subplots(1, 1, figsize=(7, 6))
    
    col_lab = data[masked, 2]
    vmin, vmax = 5, 50
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    mapper = cm.ScalarMappable(norm = norm, cmap = cmap)
    node_color = [(r, g, b) for r, g, b, a in mapper.to_rgba(col_lab)]
    sc = ax.scatter(Y_orig, all_Y_new, s = 2, c=col_lab, vmin = vmin, vmax = vmax, cmap=cmap)
    cbar = plt.colorbar(sc)
    cbar.set_label(r'SNR', fontsize = fsize)
    
    for i in range(len(all_Y_new)):
        leg += 1
        if leg == 1:
            plt.errorbar(Y_orig[i], all_Y_new[i], xerr = np.sqrt(Y_var_orig[i]), yerr = np.sqrt(all_Y_new_var[i]), fmt = 'o', capsize = 3, capthick = 2, c = node_color[i], label=' bias = {0} \n scatter = {1}'.format(bias, scatter))
        else:
            plt.errorbar(Y_orig[i], all_Y_new[i], xerr = np.sqrt(Y_var_orig[i]), yerr = np.sqrt(all_Y_new_var[i]), fmt = 'o', capsize = 3, capthick = 2, c = node_color[i])
    #plt.errorbar(Y_orig, all_Y_new, xerr = np.sqrt(Y_var_orig), yerr = np.sqrt(all_Y_new_var), fmt = 'o', c = data[masked, 2])
    plt.plot(xx, xx, color=colors[2], linestyle='--')
    plt.xlabel(r'measured {}'.format(l), size=fsize)
    plt.ylabel(r'predicted {}'.format(l), size=fsize)
    plt.tick_params(axis=u'both', direction='in', which='both')
    plt.xlim(plot_limits[l])
    plt.ylim(plot_limits[l])
    plt.tight_layout()
    plt.annotate(r'bias = {}'.format(bias) + '\n' + 'scatter = {}'.format(scatter), (5.7, 8.95), fontsize=18, bbox=dict(boxstyle="square", fc="w", ec="0.8"))
    # plt.legend(loc=4, fontsize=18, frameon=True)
    plt.title(r'$N={},\,D={},\,L={},\,Q={}$'.format(N, D-6, L, Q), fontsize = fsize)
    plt.savefig('/Users/eilers/Dropbox/projects/GPLVM_MBH/plots/1to1_{}_{}_SNR_overlap.pdf'.format(ll, name), bbox_inches = 'tight')


# fig, ax = plt.subplots(1, 1, figsize=(7, 6))
# col_lab = suc
# vmin, vmax = 0, 1
# norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
# mapper = cm.ScalarMappable(norm = norm, cmap = cmap)
# node_color = [(r, g, b) for r, g, b, a in mapper.to_rgba(col_lab)]
# sc = ax.scatter(Y_orig, all_Y_new, s = 2, c=col_lab, vmin = vmin, vmax = vmax, cmap=cmap)
# cbar = plt.colorbar(sc)
# cbar.set_label(r'SNR', fontsize = fsize)

# for i in range(len(all_Y_new)):
#     leg += 1
#     if leg == 1:
#         plt.errorbar(Y_orig[i], all_Y_new[i], xerr = np.sqrt(Y_var_orig[i]), yerr = np.sqrt(all_Y_new_var[i]), fmt = 'o', c = node_color[i], label=' bias = {0} \n scatter = {1}'.format(bias, scatter), capsize = 2)
#     else:
#         plt.errorbar(Y_orig[i], all_Y_new[i], xerr = np.sqrt(Y_var_orig[i]), yerr = np.sqrt(all_Y_new_var[i]), fmt = 'o', c = node_color[i], capsize = 2)
# #plt.errorbar(Y_orig, all_Y_new, xerr = np.sqrt(Y_var_orig), yerr = np.sqrt(all_Y_new_var), fmt = 'o', c = data[masked, 2])
# plt.plot(xx, xx, color=colors[2], linestyle='--')
# plt.xlabel(r'measured {}'.format(l), size=fsize)
# plt.ylabel(r'predicted {}'.format(l), size=fsize)
# plt.tick_params(axis=u'both', direction='in', which='both')
# plt.xlim(plot_limits[l])
# plt.ylim(plot_limits[l])
# plt.tight_layout()
# plt.legend(loc=4, fontsize=16, frameon=True)
# plt.title(r'$N={},\,D={},\,L={},\,Q={}$'.format(N, D, L, Q), fontsize = fsize)
# plt.savefig('/Users/eilers/Dropbox/projects/GPLVM_MBH/plots/1to1_{}_{}_success.pdf'.format(ll, name), bbox_inches = 'tight')

# -------------------------------------------------------------------------------
# data set
# ------------------------------------------------------------------------------- 


cmap = 'cmr.guppy_r'
fig, ax = plt.subplots(1, 1, figsize=(7, 6))

col_lab = data[:, 1]
vmin, vmax = 0, 0.2
norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
mapper = cm.ScalarMappable(norm = norm, cmap = cmap)
node_color = [(r, g, b) for r, g, b, a in mapper.to_rgba(col_lab)]
sc = ax.scatter(data[:, 0], data[:, 6], s = 2, c=col_lab, vmin = vmin, vmax = vmax, cmap=cmap)
cbar = plt.colorbar(sc)
cbar.set_label(r'redshift $z$', fontsize = fsize)

for i in range(len(all_Y_new)):
    leg += 1
    if leg == 1:
        plt.errorbar(data[i, 0], data[i, 6], yerr = data_ivar[i, 6]**(-0.5), xerr = data_ivar[i, 0]**(-0.5), fmt = 'o', capsize = 3, capthick = 2, markersize = 0, c = node_color[i]) #, label=' bias = {0} \n scatter = {1}'.format(bias, scatter))
    else:
        plt.errorbar(data[i, 0], data[i, 6], yerr = data_ivar[i, 6]**(-0.5), xerr = data_ivar[i, 0]**(-0.5), fmt = 'o', capsize = 3, capthick = 2, markersize = 0, c = node_color[i])
#plt.errorbar(Y_orig, all_Y_new, xerr = np.sqrt(Y_var_orig), yerr = np.sqrt(all_Y_new_var), fmt = 'o', c = data[masked, 2])
plt.plot(xx, xx, color=colors[2], linestyle='--')
plt.xlabel(r'$\log_{{10}}(M_\bullet / M_\odot)$', size=fsize)
plt.ylabel(r'$\log_{{10}} (L_{{\rm bol}} / \rm erg\,s^{-1})$', size=fsize)
plt.tick_params(axis=u'both', direction='in', which='both')
plt.xlim(5.8, 9.3)
plt.ylim(41, 47.2)
plt.tight_layout()
#plt.legend(loc=4, fontsize=18, frameon=True)
plt.savefig('/Users/eilers/Dropbox/projects/GPLVM_MBH/plots/sample.pdf')

# -------------------------------------------------------------------------------
# latent space
# ------------------------------------------------------------------------------- 

for qq in range(3): #N):
    f = open('files/{}_qq{}.pickle'.format(name, qq), 'rb')
    all_Y_new_i, all_Y_new_var_i, sort_r, Z_final, Z_opt_n, hyper_params, success_train, success_test = pickle.load(f)
    f.close()
    
    ind_test = np.zeros(data.shape[0], dtype = bool)
    ind_test[qq] = True
    ind_train = ~ind_test
    
    N_Z = 5 # Q
    
    fig = plt.figure(figsize = (N_Z*2.5, N_Z*3))
    widths = np.ones(N_Z)
    widths[-1] = .1
    gs = gridspec.GridSpec(N_Z, N_Z, width_ratios = widths)
    gs.update(wspace = 0., hspace = 0.)
    
    for cols in range(N_Z): 
        for rows in range(cols):
            ax = plt.subplot(gs[N_Z-1-rows, N_Z-1-cols])
            #ax.set_xlim(-0.5, 0.5)
            #ax.set_ylim(-0.5, 0.5)
            sc = ax.scatter(Z_final[:, sort_r[cols]], Z_final[:, sort_r[rows]], c = data[ind_train, 0])
            if rows == 0:
                ax.set_xlabel(r'$Z_{{\,{}}}$'.format(sort_r[cols]), fontsize = 22)
            if cols == N_Z-1:
                ax.set_ylabel(r'$Z_{{\rm{}}}$'.format(sort_r[rows]), fontsize = 22)
            if N_Z-1-cols > 0:
                ax.tick_params(labelleft = False)
            if rows > 0:
                ax.tick_params(labelbottom = False)
    
    cbar = plt.colorbar(sc, cax = plt.subplot(gs[N_Z-1, N_Z-1]))
    cbar.set_label(r'$\log_{10}(M_\bullet/M_\odot)$', fontsize = 22)
    plt.savefig('/Users/eilers/Dropbox/projects/GPLVM_MBH/plots/Zspace/Zspace_{}_qq{}.pdf'.format(name, qq), bbox_inches = 'tight')

# pick Z0 and second highest pearson R coefficient and do 2D KDE...

# -------------------------------------------------------------------------------
# predict spectra 
# ------------------------------------------------------------------------------- 

min_waves, max_waves = 1220, 5000.1 

civ = 1549.06
ciii = 1908.73 # CIII]
siiv = 1396.76
nv = 1240.14
siii_a = 1262.59
siii_b = 1304.37
oi = 1302.17
cii = 1335.30
lya = 1215.67
mgii = 2798.7
hbeta = 4861.35
oiii = 4959 #[OIII]

lines = np.array([lya, siiv, civ, ciii, mgii, hbeta])

if flexwave: 
    dline = 60.
    dlam_noline = 20.
    dlam_line = 2.
        
    wave_grid = list([min_waves]) #np.arange(max(np.round(lya) - dline, min_waves), np.round(lya) + dline, 2))
    for l in lines:
        if wave_grid[-1] > l-dline and wave_grid[-1] < l+dline and wave_grid[-1] < max_waves:
            while wave_grid[-1] > l-dline and wave_grid[-1] < l+dline:
                wave_grid.append(wave_grid[-1] + dlam_line)
        elif wave_grid[-1] < l-dline:
            while wave_grid[-1] < l-dline:
                wave_grid.append(wave_grid[-1] + dlam_noline)
            if wave_grid[-1] > l-dline and wave_grid[-1] < l+dline:
                while wave_grid[-1] > l-dline and wave_grid[-1] < l+dline and wave_grid[-1] < 5000.1:
                    wave_grid.append(wave_grid[-1] + dlam_line)
    if wave_grid[-1] < max_waves:
        while wave_grid[-1] < max_waves:          
            wave_grid.append(wave_grid[-1] + dlam_noline)
    
    wave_grid = np.array(wave_grid)
else:
    wave_grid = np.arange(min_waves, max_waves, 2)

waves = np.array(wave_grid[1:-1]) # slightly imprecise...
# waves = wave_grid[:-2] + 0.5 * np.diff(wave_grid)[:-1]
print('number of pixels: {}'.format(len(waves)))

for qq in range(25, 31): #N):
    if exists('files/{}_qq{}.pickle'.format(name, qq)):
        f = open('files/{}_qq{}.pickle'.format(name, qq), 'rb')
        all_Y_new_i, all_Y_new_var_i, sort_r, Z_final, Z_opt_n, hyper_params, success_train, success_test = pickle.load(f)
        f.close()
    
        theta_rbf, gamma_rbf = hyper_params
        theta_band, gamma_band = 1., 1.
        
        ind_test = np.zeros(N, dtype = bool)
        ind_test[qq] = True
        ind_train = np.ones(N, dtype=bool)
        ind_train[qq] = False
        
        qs = np.nanpercentile(data[ind_train, :], (2.5, 50, 97.5), axis=0)
        pivots = qs[1]
        scales = (qs[2] - qs[0]) / 4.
        scales[scales == 0] = 1.
        data_scaled = (data - pivots) / scales
        data_ivar_scaled = data_ivar * scales**2 
        
        X_input = data_scaled[ind_train, 6:]
        X_var_input = 1./data_ivar_scaled[ind_train, 6:]
        X_mask = np.ones_like(X_input).astype(bool)
        X_mask[np.isnan(X_input)] = False
        
        X_new_n, X_new_var_n, k_Z_zj = mean_var(Z_final, Z_opt_n, X_input, X_var_input, theta_rbf, theta_band, X_mask)
        
        X_new_n = X_new_n * scales[6:] + pivots[6:]
        X_new_var_n =  X_new_var_n * scales[6:]**2
    
        fig, ax = plt.subplots(1, 1, figsize = (15, 6))
        plt.plot(waves, X_new_n[1:], color = 'r', zorder = 100, lw = 2, label = 'prediction for missing spectral pixels')
        plt.plot(waves, data[ind_test, 7:][0], color = 'k', zorder = 1000, label = 'quasar spectrum in testing set')
        #plt.plot(waves, pivots[7:], color = 'b', zorder = 1000, label = 'mean quasar spectrum training set')
        # if qq < 31:
        #     plt.plot(waves, data[qq+6, 7:], color = 'b', zorder = 1000, label = 'other part of quasar spectrum (part of training set)')
        # else:
        #     plt.plot(waves, data[qq-6, 7:], color = 'b', zorder = 1000, label = 'other part of quasar spectrum (part of training set)')            
        plt.plot(waves, data_pred[qq-25, 7:], color = 'b', zorder = 1000, label = 'other part of quasar spectrum')
        plt.fill_between(waves, X_new_n[1:] - X_new_var_n[1:], X_new_n[1:] + X_new_var_n[1:], zorder = -1, color = 'r', alpha = .3)
        plt.xlim(min(waves), max(waves))
        plt.xlabel(r'rest-frame wavelength [{\AA}]', fontsize = fsize)
        plt.legend(fontsize = fsize, frameon = True)
        plt.ylim(-1, 10)
        plt.title(r'input: $\log L_{{\rm bol}} = {}\pm{},~\log M_\bullet = {}\pm{}$, output: $\log M_\bullet = {}\pm{}$'.format(np.round(data[ind_test, 6][0], 2), np.round(1./np.sqrt(data_ivar[ind_test, 6])[0], 2), np.round(data[ind_test, 0][0], 2), np.round(1./np.sqrt(data_ivar[ind_test, 0])[0], 2), np.round(all_Y_new_i[ind_test][0], 2), np.round(np.sqrt(all_Y_new_var_i[ind_test][0]), 2)), fontsize = 18)
        fig.savefig('/Users/eilers/Dropbox/projects/GPLVM_MBH/plots/spectra/spectra_prediction_{}_{}.pdf'.format(name, qq), bbox_inches = 'tight')
    

# -------------------------------------------------------------------------------
# sample latent space based on given BH masses
# ------------------------------------------------------------------------------- 

wave_grid = np.arange(min_waves, max_waves, 2)
waves = np.array(wave_grid[1:-1])

for qq in range(5):
    f = open('files/{}_qq{}.pickle'.format(name, qq), 'rb')
    all_Y_new_i, all_Y_new_var_i, sort_r, Z_final, Z_opt_n, hyper_params, success_train, success_test = pickle.load(f)
    f.close()
    
    Ax, Ay = hyper_params
    
    ind_test = np.zeros(N, dtype = bool)
    ind_test[qq] = True
    ind_train = np.ones(N, dtype=bool)
    ind_train[qq] = False
    
    qs = np.nanpercentile(data[ind_train, :], (2.5, 50, 97.5), axis=0)
    pivots = qs[1]
    scales = (qs[2] - qs[0]) / 4.
    scales[scales == 0] = 1.
    data_scaled = (data - pivots) / scales
    data_ivar_scaled = data_ivar * scales**2 
    
    np.random.seed(14573)    
    NN = 10000
    Y_new = np.zeros((NN, L))
    all_zn = np.zeros((NN, Q))
    
    spectra_MBH = np.zeros((L, 6, len(waves)))
    
    for ll in range(L):
    #ll = 0 # MBH
        if ll == 0:
            label_l = 0
        elif ll == 2:
            label_l = 1 #z
        elif ll == 1:
            label_l = 6 # Lbol
        elif ll == 3:
            label_l = 5 # lambda
        Y_input = data_scaled[ind_train, label_l]
        Y_var_input = 1./data_ivar_scaled[ind_train, label_l]
        Y_mask = np.ones_like(Y_input).astype(bool)
        Y_mask[np.isnan(Y_input)] = False
        
        X_input = data_scaled[ind_train, 7:]
        X_var_input = 1./data_ivar_scaled[ind_train, 7:]
        X_mask = np.ones_like(X_input).astype(bool)
        X_mask[np.isnan(X_input)] = False
        
        good_stars = Y_mask
    
        for n in range(NN):
            zn = np.random.normal(loc=0.0, scale=0.3, size=Q)
            all_zn[n, :] = zn
            Y_new_n, Y_new_var_n, k_Z_zj, factor = mean_var(Z_final, zn, Y_input[good_stars], Y_var_input[good_stars], Ay[ll], 1.)                
            Y_new[n, ll] = Y_new_n * scales[label_l] + pivots[label_l]
            
        #ledds = (2.36e38*10**Y_new[:, 0]) / 10**Y_new[:, 1]

    #for ll in range(3):
        if ll == 0:
            dm = 0.1
            logmasses = np.array([6, 7, 8, 9, 10])#[::-1]
        elif ll == 2:
            #dm = 0.05
            #logmasses = np.array([0, 0.25, 0.5, 0.75, 1.])
            dm = 0.01
            logmasses = np.array([0, 0.05, 0.1, 0.15, 0.2, 0.25])
        elif ll == 1:
            dm = 0.2
            logmasses = np.array([42, 43, 44, 45, 46, 47])
        elif ll == 3:
            dm = 0.02
            logmasses = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5]) #, 0.8])
        
        for i, mi in enumerate(logmasses):
            mask = np.abs(Y_new[:, ll] - logmasses[i]) < dm
            # if ll < 2:
            #     mask = np.abs(Y_new[:, ll] - logmasses[i]) < dm
            # else:
            #     mask = np.abs(ledds - logmasses[i]) < dm                
            # print(mi, np.sum(mask))
            # if ll == 0: #fix Lbol
            #     mask *= np.abs(Y_new[:, 1] - np.mean(Y_new[:, 1])) < 1
            # if ll == 1: #fix MBH
            #     mask *= np.abs(Y_new[:, 0] - np.mean(Y_new[:, 0])) < 0.5
            spec = np.zeros((np.sum(mask), len(waves)))
            print(mi, np.sum(mask))
            for zi in range(np.sum(mask)):
                X_new_n, X_new_var_n, k_Z_zj = mean_var(Z_final, all_zn[mask][zi], X_input, X_var_input, Ax, 1., X_mask)
                X_new_n = X_new_n * scales[7:] + pivots[7:]
                X_new_var_n =  X_new_var_n * scales[7:]**2
                spec[zi, :] = X_new_n
            spectra_MBH[ll, i, :] = np.percentile(spec, 50, axis = 0)
            #spectra_MBH_16[i, :] = np.percentile(spec, 16, axis = 0)
            #spectra_MBH_84[i, :] = np.percentile(spec, 84, axis = 0)

    #colors = sns.color_palette("ch:start=.2,rot=-.3")[::-1]
    #colors = sns.color_palette("Spectral")
    colors = sns.color_palette("cmr.ocean")[::-1]
   
    # plt.close()    
    # fig, ax = plt.subplots(3, 1, figsize = (12, 11), sharex = True)
    # plt.subplots_adjust(hspace = 0)
    
    fig = plt.figure(figsize = (12, 13))
    gs = gridspec.GridSpec(L, 1)
    gs.update(hspace=0.0, wspace = 0.0)
    #axb = plt.axes([0.735, 0.66, 0.15, 0.2])
            
    for ll in range(L): 
        ax = plt.subplot(gs[ll])        
        if ll == 0:
            dm = 0.1
            logmasses = np.array([6, 7, 8, 9, 10])#[::-1]
            # axb = inset_axes(ax, width="30%", height="75%",
            #        bbox_to_anchor=(.2, .32, .6, .5),
            #        bbox_transform=ax.transAxes, loc=3)
        elif ll == 2:
            dm = 0.01
            logmasses = np.array([0, 0.05, 0.1, 0.15, 0.2, 0.25])
            #dm = 0.05
            #logmasses = np.array([0, 0.25, 0.5, 0.75, 1.])
        elif ll == 1:
            dm = 0.2
            logmasses = np.array([42, 43, 44, 45, 46, 47])
            axb = inset_axes(ax, width="30%", height="75%",
                   bbox_to_anchor=(.47, .4, .6, .5),
                   bbox_transform=ax.transAxes, loc=3)
        elif ll == 3:
            dm = 0.02
            logmasses = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5]) #, 0.8])
    
        for mi in range(len(logmasses)):
            if ll == 0:
                ax.plot(waves, spectra_MBH[ll, mi, :], label = r'$\log_{{10}}(M_\bullet/M_\odot) = {}$'.format(int(logmasses[mi])), lw = 2, color = colors[mi+1], zorder = 10)
                # axb.plot(waves, spectra_MBH[ll, mi+1, :], lw = 2, color = colors[mi+1])
                # axb.set_ylim(1.2, 3.)
                # axb.set_xlim(1442, 1520)
                # axb.text(1445, 2.6, 'A', fontsize = 14)
                # axb.tick_params(axis=u'both', direction='in', which='both', labelleft = False)
            elif ll == 2:
                ax.plot(waves, spectra_MBH[ll, mi, :], label = r'$z = {}$'.format(logmasses[mi]), lw = 2, color = colors[mi])
            elif ll == 1:
                ax.plot(waves, spectra_MBH[ll, mi, :], label = r'$\log_{{10}}(L_{{\rm bol}}/\rm erg\,s^{{-1}}) = {}$'.format(logmasses[mi]), lw = 2, color = colors[mi])                
                axb.plot(waves, spectra_MBH[ll, mi, :], lw = 2, color = colors[mi])
                axb.set_ylim(1.8, 3.2)
                axb.set_xlim(1620, 1655) 
                # axb.text(1623, 2.95, 'B', fontsize = 14)
                axb.tick_params(axis=u'both', direction='in', which='both', labelleft = False)
            elif ll == 3:
                 ax.plot(waves, spectra_MBH[ll, mi, :], label = r'$\lambda_{{\rm Edd}} = {}$'.format(logmasses[mi]), lw = 2, color = colors[mi])
        ax.legend(fontsize = 15, loc = 1)
        ax.set_ylim(0, 14.9)
        ax.set_xlim(1220, 2000)
        ax.axvline(civ, linestyle = '--', color = '#aeb6bf', lw = 1.5, zorder = -10) 
        ax.text(civ-5, 13., r'C\,IV', fontsize = 12, rotation=90, color = '#aeb6bf', bbox=dict(boxstyle="square", fc="w", ec="w"))                    
        ax.axvline(1639.2, linestyle = '--', color = '#aeb6bf', lw = 1.5, zorder = -10) 
        ax.text(1639.2-5, 13, r'He\,II', fontsize = 12, rotation=90, color = '#aeb6bf', bbox=dict(boxstyle="square", fc="w", ec="w"))                    
        
        ax.axvline(1239.5, linestyle = '--', color = '#aeb6bf', lw = 1.5, zorder = -10) 
        ax.text(1239.5-5, 13, r'N\,V', fontsize = 12, rotation=90, color = '#aeb6bf', bbox=dict(boxstyle="square", fc="w", ec="w"))                    

        ax.axvline(1334.0 , linestyle = '--', color = '#aeb6bf', lw = 1.5, zorder = -10) 
        ax.text(1334.0 -5, 13, r'C\,II', fontsize = 12, rotation=90, color = '#aeb6bf', bbox=dict(boxstyle="square", fc="w", ec="w"))                    

        ax.axvline(1302.1  , linestyle = '--', color = '#aeb6bf', lw = 1.5, zorder = -10) 
        ax.text(1302.1  -5, 12, r'Si\,II+O\,I', fontsize = 12, rotation=90, color = '#aeb6bf', bbox=dict(boxstyle="square", fc="w", ec="w"))                    

        ax.axvline(1401.2 , linestyle = '--', color = '#aeb6bf', lw = 1.5, zorder = -10) 
        ax.text(1401.2 -5, 11, r'Si\,IV+O\,IV]', fontsize = 12, rotation=90, color = '#aeb6bf', bbox=dict(boxstyle="square", fc="w", ec="w"))                    

        ax.axvline(1889.5 , linestyle = '--', color = '#aeb6bf', lw = 1.5, zorder = -10) 
        ax.text(1889.5 -5, 3.8, r'Si\,III]', fontsize = 12, rotation=90, color = '#aeb6bf', bbox=dict(boxstyle="square", fc="w", ec="w"))                    

        ax.axvline(1908.7, linestyle = '--', color = '#aeb6bf', lw = 1.5, zorder = -10)  #CIII]
        ax.text(1908.7-5, 4, r'C\,III]', fontsize = 12, rotation=90, color = '#aeb6bf', bbox=dict(boxstyle="square", fc="w", ec="w"))                    
        if ll == 2:
            ax.tick_params(axis=u'both', direction='in', which='both')
            ax.set_xlabel('rest-frame wavelength [{\AA}]', fontsize = fsize)
        else:
            ax.tick_params(axis=u'both', direction='in', which='both', labelbottom = False)            
        if ll == 1:
            ax.set_ylabel('normalized flux', fontsize = fsize)
    fig.savefig('/Users/eilers/Dropbox/projects/GPLVM_MBH/plots/pred_new/spectra_{}_qq{}_fixed.pdf'.format(name, qq), bbox_inches = 'tight')

    # if ll == 0:
    #     fig.savefig('/Users/eilers/Dropbox/projects/GPLVM_MBH/plots/pred_new/spectraMBH_{}_qq{}.pdf'.format(name, qq), bbox_inches = 'tight')
    # elif ll == 1:
    #     fig.savefig('/Users/eilers/Dropbox/projects/GPLVM_MBH/plots/pred_new/spectraz_{}_qq{}.pdf'.format(name, qq), bbox_inches = 'tight')
    # elif ll == 2:
    #     fig.savefig('/Users/eilers/Dropbox/projects/GPLVM_MBH/plots/pred_new/spectraLbol_{}_qq{}.pdf'.format(name, qq), bbox_inches = 'tight')
    # elif ll == 3:
    #     fig.savefig('/Users/eilers/Dropbox/projects/GPLVM_MBH/plots/pred_new/spectraalpha_{}_qq{}.pdf'.format(name, qq), bbox_inches = 'tight')


# -------------------------------------------------------------------------------'''
# averaging predictions
# ------------------------------------------------------------------------------- 


'''N = 64
L = 1
Q = 16
D = 1890 #
f = open('data_HST_SDSS_bin2_5000.pickle', 'rb')
data, data_ivar = pickle.load(f)
f.close()


N_runs = 3*3*3
params = np.ones((N_runs, 3))
params[0, :] = np.array([0.5, 0.5, 0.5])
params[1, :] = np.array([0.5, 0.5, 1])
params[2, :] = np.array([0.5, 1, 0.5])
params[3, :] = np.array([1, 0.5, 0.5])
params[4, :] = np.array([0.5, 1, 1])
params[5, :] = np.array([1, 0.5, 1])
params[6, :] = np.array([1, 1, 0.5])
params[7, :] = np.array([1, 1, 1])
params[8, :] = np.array([1, 1, 2])
params[9, :] = np.array([1, 2, 1])
params[10, :] = np.array([2, 1, 1])
params[11, :] = np.array([2, 0.5, 1])
params[12, :] = np.array([1, 2, 2])
params[13, :] = np.array([2, 1, 2])
params[14, :] = np.array([2, 2, 1])
params[15, :] = np.array([0.5, 0.5, 2])
params[16, :] = np.array([0.5, 2, 0.5])
params[17, :] = np.array([2, 0.5, 0.5])
params[18, :] = np.array([0.5, 2, 2])
params[19, :] = np.array([2, 0.5, 2])
params[20, :] = np.array([2, 2, 0.5])
params[21, :] = np.array([2, 2, 2])
params[22, :] = np.array([0.5, 1, 2])
params[23, :] = np.array([0.5, 2, 1])
params[24, :] = np.array([1, 2, 0.5])
params[25, :] = np.array([2, 1, 0.5])
params[26, :] = np.array([1, 0.5, 2])

all_Y_new = np.zeros((N+1, N_runs))
all_Y_new_var = np.zeros((N+1, N_runs))

b1, b2, b3 = 0.5, 0.5, 0.5

for nn in range(N_runs):  
    
    b1, b2, b3 = params[nn]
    
    if b1 > 0.5: b1 = int(b1)
    if b2 > 0.5: b2 = int(b2)
    if b3 > 0.5: b3 = int(b3)
    
    name = 'ps_s42_hyper1D_HST_SDSS_bin2_5000_fixedband_{}_{}_{}_newps_mbhcorrect_cross_N{}_D{}_L{}_Q{}'.format(b1, b2, b3, N, D, L, Q)

    for i in range(N+1):
        if exists('files/f_{}_qq{}.pickle'.format(name, i)):
            f = open('files/f_{}_qq{}.pickle'.format(name, i), 'rb')
            all_Y_new_i, all_Y_new_var_i, sort_r, Z_final, Z_opt_n, hyper_params, success_train, success_test = pickle.load(f)
            f.close()
            all_Y_new[i, nn] = all_Y_new_i[i, :][0]
            all_Y_new_var[i, nn] = all_Y_new_var_i[i, :][0]
        else:
            print('{}, file missing for N={}!'.format(nn, i+1))
             
all_Y_new[all_Y_new == 0.] = np.nan    

# black holes
masked_orig = data[:, 2] > 5
data = data[masked_orig]
data_ivar = data_ivar[masked_orig]
l = labels[0]
Y_orig = data[:, 0]
Y_var_orig = 1./data_ivar[:, 0] 


#mean_Y = np.nanmean(all_Y_new, axis = 1)
mean_Y = np.zeros(N+1)
for i in range(N+1):
    mask_i = np.isfinite(all_Y_new[i, :])
    mean_Y[i] = np.average(all_Y_new[i, mask_i], weights = 1./all_Y_new_var[i, mask_i])
mean_Y_var = np.nanmean(all_Y_new_var, axis = 1)
   

scatter = np.round(np.std(Y_orig - mean_Y), 4)
bias = np.round(np.mean(Y_orig - mean_Y), 4)    
chi2_label = np.round(np.sum((Y_orig - mean_Y)**2 / (mean_Y_var + Y_var_orig)) / N, 4) # 0.3**2 from unkown geometry factor

xx = [-1000, 1000]
cdict = {1.: colors[0], 2.: colors[1], 5.: colors[5]}
leg = 0
cmap = 'viridis'
fig, ax = plt.subplots(1, 1, figsize=(7, 6))

col_lab = data[:, 2]
vmin, vmax = 5, 50
norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
mapper = cm.ScalarMappable(norm = norm, cmap = cmap)
node_color = [(r, g, b) for r, g, b, a in mapper.to_rgba(col_lab)]
sc = ax.scatter(Y_orig, mean_Y, s = 2, c=col_lab, vmin = vmin, vmax = vmax, cmap=cmap)
cbar = plt.colorbar(sc)
cbar.set_label(r'SNR', fontsize = fsize)


for i in range(len( all_Y_new)):
    leg += 1
    if leg == 1:
        plt.errorbar(Y_orig[i], mean_Y[i], xerr = np.sqrt(Y_var_orig[i]), yerr = np.sqrt(mean_Y_var[i]), fmt = 'o', c = node_color[i], label=' bias = {0} \n scatter = {1} \n reduced $\chi^2$ = {2}'.format(bias, scatter, chi2_label))
    else:
        plt.errorbar(Y_orig[i], mean_Y[i], xerr = np.sqrt(Y_var_orig[i]), yerr = np.sqrt(mean_Y_var[i]), fmt = 'o', c = node_color[i])
#plt.errorbar(Y_orig, all_Y_new, xerr = np.sqrt(Y_var_orig), yerr = np.sqrt(all_Y_new_var), fmt = 'o', c = data[masked, 2])
plt.plot(xx, xx, color=colors[2], linestyle='--')
plt.xlabel(r'measured labels {}'.format(l), size=fsize)
plt.ylabel(r'inferred values {}'.format(l), size=fsize)
plt.tick_params(axis=u'both', direction='in', which='both')
plt.xlim(plot_limits[l])
plt.ylim(plot_limits[l])
plt.tight_layout()
plt.legend(loc=4, fontsize=16, frameon=True)
plt.title(r'$N={},\,D={},\,L={},\,Q={}$'.format(N+1, D, L, Q), fontsize = fsize)
plt.savefig('/Users/eilers/Dropbox/projects/RM_black_hole_masses/plots/new/1to1_all_runs_weighted_averaged_SNR.pdf', bbox_inches = 'tight')



# -------------------------------------------------------------------------------
# predicting spectra for given black hole mass...
# ------------------------------------------------------------------------------- 

min_waves, max_waves = 1220, 3000.1 
wave_grid = np.arange(min_waves, max_waves, 2)
waves = wave_grid[:-2] + 0.5 * np.diff(wave_grid)[:-1]

# min_waves, max_waves = 1220, 5100 

# civ = 1549.06
# ciii = 1908.73 # CIII]
# siiv = 1396.76
# nv = 1240.14
# siii_a = 1262.59
# siii_b = 1304.37
# oi = 1302.17
# cii = 1335.30
# lya = 1215.67
# mgii = 2798.7
# hbeta = 4861.35
# oiii = 4959 #[OIII]

# dline = 60.

# dlam_noline = 10.
# dlam_line = 2.

# lines = np.array([lya, nv, siii_a, oi, siii_b, cii, siiv, civ, ciii, mgii, hbeta])


# wave_grid = list(np.arange(max(np.round(lya) - dline, min_waves), np.round(lya) + dline, 1))
# for l in lines[1:]:
#     if wave_grid[-1] > l-dline and wave_grid[-1] < l+dline and wave_grid[-1] < max_waves:
#         while wave_grid[-1] > l-dline and wave_grid[-1] < l+dline:
#             wave_grid.append(wave_grid[-1] + dlam_line)
#     elif wave_grid[-1] < l-dline:
#         while wave_grid[-1] < l-dline:
#             wave_grid.append(wave_grid[-1] + dlam_noline)
#         if wave_grid[-1] > l-dline and wave_grid[-1] < l+dline:
#             while wave_grid[-1] > l-dline and wave_grid[-1] < l+dline and wave_grid[-1] < max_waves:
#                 wave_grid.append(wave_grid[-1] + dlam_line)
# if wave_grid[-1] < max_waves:
#     while wave_grid[-1] < max_waves:          
#         wave_grid.append(wave_grid[-1] + dlam_noline)

# waves = np.array(wave_grid[1:-1])
# print('number of pixels: {}'.format(len(waves)))

for i in range(N+1):
    if exists('files/f_{}_qq{}.pickle'.format(name, i)):
        f = open('files/f_{}_qq{}.pickle'.format(name, i), 'rb')
        all_Y_new_i, all_Y_new_var_i, sort_r, Z_final, Z_opt_n, hyper_params, success_train, success_test = pickle.load(f)
        f.close()
    
        f = open('data_HST_bin2_3000.pickle', 'rb')
        data, data_ivar = pickle.load(f)
        f.close()
        if i == 49: 
            xx = fits.open('BOSS/Mrk290.fits')
            data_xx = xx[1].data
            header = xx[0].header
            wave_xx = 10**(header['COEFF0'] + header['COEFF1'] * np.arange(len(data_xx['flux'])))
            wave_xx = wave_xx / (1+data[i, 1])
    
        theta_band, gamma_band, theta_rbf, gamma_rbf = hyper_params
        
        ind_test = np.zeros(N+1, dtype = bool)
        ind_test[i] = True
        ind_train = np.ones(N+1, dtype=bool)
        ind_train[i] = False
        
        X_input = data[ind_train, 6:]
        X_var_input = 1./data_ivar[ind_train, 6:]
        X_mask = np.ones_like(X_input).astype(bool)
        X_mask[np.isnan(X_input)] = False
        
        X_new_n, X_new_var_n, k_Z_zj = mean_var(Z_final, Z_opt_n, X_input, X_var_input, theta_rbf, theta_band, X_mask)
    
        fig, ax = plt.subplots(1, 1, figsize = (15, 6))
        plt.plot(waves, X_new_n[1:], color = 'r', zorder = 100, lw = 2, label = 'prediction for missing spectral pixels')
        plt.plot(waves, data[ind_test, 7:][0], color = 'k', zorder = 1000, label = 'quasar spectrum in testing set')
        if i == 49: 
            plt.plot(wave_xx, data_xx['flux']/ 1000, color = 'b')
        plt.fill_between(waves, X_new_n[1:] - X_new_var_n[1:], X_new_n[1:] + X_new_var_n[1:], zorder = -1, color = 'r', alpha = .3)
        plt.xlim(min(waves), max(waves))
        plt.xlabel(r'rest-frame wavelength [{\AA}]', fontsize = fsize)
        plt.legend(fontsize = fsize, frameon = True)
        plt.title(r'input: $\log L_{{\rm bol}} = {}\pm{},~\log M_\bullet = {}\pm{}$, output: $\log L_{{\rm bol}} = {}\pm{},~\log M_\bullet = {}\pm{}$'.format(np.round(data[ind_test, 6][0], 2), np.round(1./np.sqrt(data_ivar[ind_test, 6])[0], 2), np.round(data[ind_test, 0][0], 2), np.round(1./np.sqrt(data_ivar[ind_test, 0])[0], 2), np.round(X_new_n[0], 2), np.round(np.sqrt(X_new_var_n[0]), 2), np.round(all_Y_new_i[ind_test][0][0], 2), np.round(np.sqrt(all_Y_new_var_i[ind_test][0][0]), 2)), fontsize = 18)
        fig.savefig('/Users/eilers/Dropbox/projects/RM_black_hole_masses/plots/new/spectra_prediction_{}_{}.pdf'.format(name, i), bbox_inches = 'tight')
    


i = 0
Z_new_n = np.zeros((3, Q))
for j, m in enumerate(range(5, 10, 2)):
    print('log M =', m)
    Y_new = np.array([m])
    closest_qso = np.argmin(np.abs(data[:, 0] - Y_new))
    
    f = open('files/f_{}_qq{}.pickle'.format(name, i), 'rb')
    all_Y_new_i, all_Y_new_var_i, sort_r, Z_final, Z_opt_n, hyper_params, success_train, success_test = pickle.load(f)
    f.close()
    
    f = open('data_HST_bin2_3000.pickle', 'rb')
    data, data_ivar = pickle.load(f)
    f.close()
    
    qs = np.nanpercentile(data, (2.5, 50, 97.5), axis=0)
    pivots = qs[1]
    scales = (qs[2] - qs[0]) / 4.
    
    Y_new = (Y_new - pivots[0]) /scales[0]
    
    data_scaled = (data - pivots) / scales
    data_ivar_scaled = data_ivar * scales**2 
    
    theta_band, gamma_band, theta_rbf, gamma_rbf = hyper_params
    
    ind_test = np.zeros(N+1, dtype = bool)
    ind_test[i] = True
    ind_train = np.ones(N+1, dtype=bool)
    ind_train[i] = False
    
    Y = data_scaled[ind_train, 0]
    Y_var = 1./data_ivar_scaled[ind_train, 0]
    
    X_input = data_scaled[ind_train, 6:]
    X_var_input = 1./data_ivar_scaled[ind_train, 6:]
    X_mask = np.ones_like(X_input).astype(bool)
    X_mask[np.isnan(X_input)] = False
    
    z0 = Z_opt_n
    
    samples = predictX(Y_new, Y[:, None], Y_var[:, None], Z_final, hyper_params, z0, name)
     
    #fig = corner.corner(samples8, quantiles = [0.16, 0.5, 0.84])   
    Z25, Z16, Z_new, Z84, Z975 = np.percentile(samples, (2.5, 16, 50, 84, 97.5), axis = 0)    
    Z_new_n[j, :] = Z_new

#Y_new_pred = Y_new * scales[inds_label] + pivots[inds_label]

np.random.seed(42)

fig, ax = plt.subplots(1, 1, figsize = (15, 6))
for j, m in enumerate(range(5, 10, 2)):
    X_new_n, X_new_var_n, k_Z_zj = mean_var(Z_final, Z_new_n[j, :], X_input, X_var_input, theta_rbf, theta_band, X_mask)
    X_new_mean = X_new_n * scales[6:] + pivots[6:]
    ax.plot(waves, X_new_mean[1:], color = colors[j], lw = 1, zorder = 10, label = r'$\log (M_\bullet/M_\odot) = {}$'.format(m))

#ax.plot(waves, X_new_mean7[1:], color = colors[0], lw = 1, zorder = 10, label = r'$\log (M_\bullet/M_\odot) = 7$')
#ax.plot(waves, X_new_mean8[1:], color = colors[1], lw = 1, zorder = 10, label = r'$\log (M_\bullet/M_\odot) = 8$')
#ax.plot(waves, X_new_mean[1:], color = colors[2], lw = 1, zorder = 10, label = r'$\log (M_\bullet/M_\odot) = 9$')
ax.legend(fontsize = fsize)
ax.set_xlim(min(waves), max(waves))
# for i in range(20):  
#     n = np.random.choice(samples9.shape[0])
#     X_new_n, X_new_var_n, k_Z_zj = mean_var(Z_final, samples9[n, :], X_input, X_var_input, theta_rbf, theta_band, X_mask)
#     X_new_n = X_new_n * scales[6:] + pivots[6:]
#     ax.plot(waves, X_new_n[1:], color = colors[2], lw = .8, alpha = .5)
#     #X_new_n, X_new_var_n, k_Z_zj = mean_var(Z_final, samples8[n, :], X_input, X_var_input, theta_rbf, theta_band, X_mask)
#     #X_new_n = X_new_n * scales[6:] + pivots[6:]
#     #ax.plot(waves, X_new_n[1:], color = 'r', lw = .8, alpha = .3)
fig.savefig('/Users/eilers/Dropbox/projects/RM_black_hole_masses/plots/new/given_masses_{}_{}.pdf'.format(name, i), bbox_inches = 'tight')

# -------------------------------------------------------------------------------
# latent space
# ------------------------------------------------------------------------------- 

NNN = Q

fig = plt.figure(figsize = (24, 30))
widths = np.ones(NNN)
widths[-1] = .1
gs = gridspec.GridSpec(NNN, NNN, width_ratios = widths)
gs.update(wspace = 0., hspace = 0.)

for cols in range(NNN):   
    for rows in range(cols):
        #print(sort_r[cols], sort_r[rows])
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

# -------------------------------------------------------------------------------
# new SDSS DR7 
# ------------------------------------------------------------------------------- 

name = 'SDSSDR7_train100_flexwave_fixedband_1_1_1'
f = open('files/f_ps_s42_hyper1D_{}_newps_mbhcorrect_cross_N100_D357_L1_Q16_qq0.pickle'.format(name), 'rb')
Y_new_test, Y_new_var_test, Z_new_test, data_test, data_ivar_test = pickle.load(f)
f.close()

labels = list(['$\log_{10}\,(M_{\\bullet}/M_\odot)$', '$\log_{10}\,L_{\\rm bol}$']) #, '$z$'])
plot_limits = {}
plot_limits['$\log_{10}\,(M_{\\bullet}/M_\odot)$'] = (6, 11)
plot_limits['$z$'] = (0, 2)
plot_limits['$\log_{10}\,L_{\\rm bol}$'] = (40, 50)

fig, ax = plt.subplots(1, 1, figsize=(7, 6))
l = labels[0]
xx = [-1000, 1000]
plt.scatter(data_test, Y_new_test)
plt.plot(xx, xx, color=colors[2], linestyle='--')
plt.xlabel(r'measured labels {}'.format(l), size=fsize)
plt.ylabel(r'inferred values {}'.format(l), size=fsize)
plt.tick_params(axis=u'both', direction='in', which='both')
plt.xlim(plot_limits[l])
plt.ylim(plot_limits[l])
plt.tight_layout()
#plt.legend(loc=4, fontsize=16, frameon=True)
fig.savefig('/Users/eilers/Dropbox/projects/RM_black_hole_masses/plots/new/1to1_{}.pdf'.format(name), bbox_inches = 'tight')


# print('prediction for test objects...')
    
    # Y_new_test = np.zeros((np.sum(ind_test), L))
    # Y_new_var_test = np.zeros((np.sum(ind_test), L))
    # Z_new_test = np.zeros((np.sum(ind_test), Q))
    # N_new = np.sum(ind_test) #1 
    
    # X_new = X[ind_test, :] 
    # X_var_new = X_var[ind_test, :]
    
    # X_mask_new = np.ones_like(X_new).astype(bool)
    # X_mask_new[np.isnan(X_new)] = False
    
    # chi2 = Chi2_Matrix(X_input, 1./X_var_input, X_new, 1./X_var_new)
    # all_NN = np.zeros((np.sum(ind_test), L))
    
    # all_chis = []
    
    # for n in range(N_new):
        
    #     # starting_guess
    #     y0, index_n = NN(n, chi2, Y_input)
    #     z0 = Z_final[index_n, :]
    #     all_NN[n, :] = y0
        
    #     #Z_opt_n, success_z, samples = predictY(X_new[n, :], X_var_new[n, :], X_input, X_var_input, Y_input, Y_var_input, Z_final, hyper_params, y0, z0, qq, name, X_mask, Y_mask, X_mask_new[n, :])
    #     Z_opt_n, success_z = predictY(X_new[n, :], X_var_new[n, :], X_input, X_var_input, Y_input, Y_var_input, Z_final, hyper_params, y0, z0, qq, name, X_mask, Y_mask, X_mask_new[n, :])
        
    #     #Z25, Z16, Z_new_n, Z84, Z975 = np.percentile(samples, (2.5, 16, 50, 84, 97.5), axis = 0)    
    #     print('optimized latents: ', Z_opt_n)

# n_pix = np.zeros((N+1))
# for i in range(N+1):
#     n_pix[i] = np.sum(np.isfinite(data[i, 6:]))

# fig, ax = plt.subplots(1, 1, figsize=(6, 5))
# plt.scatter(n_pix, np.sqrt(all_Y_new_var))
# plt.xlabel(r'$N_{\rm pix}$', size=fsize)
# plt.ylabel(r'$\sigma_{y_\star}$', size=fsize)
# plt.title(r'$N={},\,D={},\,L={},\,Q={}$'.format(N+1, D, L, Q), fontsize = fsize)
# plt.savefig('/Users/eilers/Dropbox/projects/RM_black_hole_masses/plots/new/sigy_vs_Npix_{}.pdf'.format(name), bbox_inches = 'tight')

# fig, ax = plt.subplots(1, 1, figsize=(6, 5))
# plt.scatter(n_pix, np.sqrt(all_Y_new_var))
# plt.xlabel(r'$N_{\rm pix}$', size=fsize)
# plt.ylabel(r'$\sigma_{y_\star}$', size=fsize)
# plt.title(r'$N={},\,D={},\,L={},\,Q={}$'.format(N+1, D, L, Q), fontsize = fsize)
# plt.savefig('/Users/eilers/Dropbox/projects/RM_black_hole_masses/plots/new/sigy_vs_Npix_{}.pdf'.format(name), bbox_inches = 'tight')

for i in range(N+1):
    ind_train = np.ones(N+1, dtype = bool)   
    if exists('files/f_{}_qq{}.pickle'.format(name, i)):
        f = open('files/f_{}_qq{}.pickle'.format(name, i), 'rb')
        sort_r, all_Y_new_i, all_Y_new_var_i, samples, Z_final = pickle.load(f)
        f.close()
        
        ind_train[i] = False
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
                
        plt.savefig('plots/new/corners/corner_Z_{0}_{1}.pdf'.format(name, i), bbox_inches = 'tight') 
        plt.close()

# -------------------------------------------------------------------------------'''

    