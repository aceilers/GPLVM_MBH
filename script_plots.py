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
#import cmasher as cmr
from matplotlib import cm
import matplotlib.gridspec as gridspec
from astropy.io import fits
import itertools

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

labels = list(['$\log_{10}\,(M_{\\bullet}/M_\odot)$', '$\log_{10}\,L_{\\rm bol}$']) #, '$z$'])
plot_limits = {}
plot_limits['$\log_{10}\,(M_{\\bullet}/M_\odot)$'] = (6, 9.5)
plot_limits['$z$'] = (0, 2)
plot_limits['$\log_{10}\,L_{\\rm bol}$'] = (40, 50)

h1, h2, h3 = 1, 1, 1
flexwave = False
HST_only = True

if flexwave:
    f = open('data_HST_SDSS_flexwave2_1220_5000.pickle', 'rb')
else:
    f = open('data_HST_SDSS_1220_5000_3A.pickle', 'rb')
if HST_only:
    f = open('data_HST_1220_5000_3A.pickle', 'rb')
data, data_ivar = pickle.load(f)
f.close()

N = data.shape[0]
D = data.shape[1]
L = 1
Q = 16


if flexwave: 
    name = 'f_ps_s42_HST_SDSS_flexwave2_1220_5000_fixedband_{}_{}_{}_noSNRcut_cross_N75_D460_L1_Q16'.format(h1, h2, h3)
else:
    name = 'f_ps_s42_HST_SDSS_1220_5000_3A_fixedband_{}_{}_{}_noSNRcut_cross_N75_D1260_L1_Q16'.format(h1, h2, h3)
if HST_only:
    name = 'f_ps_s42_HST_1220_5000_3A_fixedband_{}_{}_{}_noSNRcut_cross_N30_D1260_L1_Q16'.format(h1, h2, h3)

all_Y_new = np.zeros((N, L))
all_Y_new_var = np.zeros((N, L))

for qq in range(N):
    if exists('files/{}_qq{}.pickle'.format(name, qq)):
        f = open('files/{}_qq{}.pickle'.format(name, qq), 'rb')
        all_Y_new_i, all_Y_new_var_i, sort_r, Z_final, Z_opt_n, hyper_params, success_train, success_test = pickle.load(f)
        f.close()
        print(success_train, success_test)
        all_Y_new[qq, :] = all_Y_new_i[qq, :]
        all_Y_new_var[qq, :] = all_Y_new_var_i[qq, :]
    else:
        print('file missing for N={}!'.format(qq))
        

# black holes
l = labels[0]
Y_orig = data[:, 0]
Y_var_orig = 1./data_ivar[:, 0]
all_Y_new = all_Y_new[:, 0]
all_Y_new_var = all_Y_new_var[:, 0]

# # Lbol
# l = labels[1]
# Y_orig = data[:, 6]
# Y_var_orig = 1./data_ivar[:, 6]
# all_Y_new = all_Y_new[:, 1]
# all_Y_new_var = all_Y_new_var[:, 1]

masked = all_Y_new > 0
all_Y_new = all_Y_new[masked] 
all_Y_new_var = all_Y_new_var[masked]
masked_orig = data[:, 2] > 0 #5
data = data[masked_orig]
Y_orig = Y_orig[masked_orig][masked]  
Y_var_orig = Y_var_orig[masked_orig][masked]    

scatter = np.round(np.std(Y_orig - all_Y_new), 4)
bias = np.round(np.mean(Y_orig - all_Y_new), 4)    
chi2_label = np.round(np.sum((Y_orig - all_Y_new)**2 / (all_Y_new_var + Y_var_orig)) / N, 4) # 0.3**2 from unkown geometry factor

xx = [-1000, 1000]
cdict = {1.: colors[0], 2.: colors[1], 5.: colors[5]}
leg = 0
cmap = 'viridis'
fig, ax = plt.subplots(1, 1, figsize=(7, 6))

# for g in np.unique(data[masked, 4]):
#     leg += 1
#     ix = np.where(data[masked, 4] == g)
#     if leg == 1: 
#         plt.errorbar(Y_orig[ix], all_Y_new[ix], xerr = np.sqrt(Y_var_orig[ix]), yerr = np.sqrt(all_Y_new_var[ix]), fmt = 'o', c=cdict[g], label=' bias = {0} \n scatter = {1} \n reduced $\chi^2$ = {2}'.format(bias, scatter, chi2_label))
#     else:
#         plt.errorbar(Y_orig[ix], all_Y_new[ix], xerr = np.sqrt(Y_var_orig[ix]), yerr = np.sqrt(all_Y_new_var[ix]), fmt = 'o', c=cdict[g])

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
        plt.errorbar(Y_orig[i], all_Y_new[i], xerr = np.sqrt(Y_var_orig[i]), yerr = np.sqrt(all_Y_new_var[i]), fmt = 'o', c = node_color[i], label=' bias = {0} \n scatter = {1}'.format(bias, scatter))
    else:
        plt.errorbar(Y_orig[i], all_Y_new[i], xerr = np.sqrt(Y_var_orig[i]), yerr = np.sqrt(all_Y_new_var[i]), fmt = 'o', c = node_color[i])
#plt.errorbar(Y_orig, all_Y_new, xerr = np.sqrt(Y_var_orig), yerr = np.sqrt(all_Y_new_var), fmt = 'o', c = data[masked, 2])
plt.plot(xx, xx, color=colors[2], linestyle='--')
plt.xlabel(r'measured {}'.format(l), size=fsize)
plt.ylabel(r'predicted {}'.format(l), size=fsize)
plt.tick_params(axis=u'both', direction='in', which='both')
plt.xlim(plot_limits[l])
plt.ylim(plot_limits[l])
plt.tight_layout()
plt.legend(loc=4, fontsize=16, frameon=True)
plt.title(r'$N={},\,D={},\,L={},\,Q={}$'.format(N, D, L, Q), fontsize = fsize)
plt.savefig('/Users/eilers/Dropbox/projects/RM_black_hole_masses/plots/new/1to1_{}_SNR.pdf'.format(name), bbox_inches = 'tight')


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
    plt.savefig('/Users/eilers/Dropbox/projects/RM_black_hole_masses/plots/new/Zspace/Zspace_{}_qq{}.pdf'.format(name, qq), bbox_inches = 'tight')

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
    wave_grid = np.arange(min_waves, max_waves, 3)

waves = np.array(wave_grid[1:-1]) # slightly imprecise...
# waves = wave_grid[:-2] + 0.5 * np.diff(wave_grid)[:-1]
print('number of pixels: {}'.format(len(waves)))

for qq in range(5): #N):
    if exists('files/{}_qq{}.pickle'.format(name, qq)):
        f = open('files/{}_qq{}.pickle'.format(name, qq), 'rb')
        all_Y_new_i, all_Y_new_var_i, sort_r, Z_final, Z_opt_n, hyper_params, success_train, success_test = pickle.load(f)
        f.close()
    
        theta_band, gamma_band, theta_rbf, gamma_rbf = hyper_params
        
        ind_test = np.zeros(N, dtype = bool)
        ind_test[i] = True
        ind_train = np.ones(N, dtype=bool)
        ind_train[i] = False
        
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
        plt.fill_between(waves, X_new_n[1:] - X_new_var_n[1:], X_new_n[1:] + X_new_var_n[1:], zorder = -1, color = 'r', alpha = .3)
        plt.xlim(min(waves), max(waves))
        plt.xlabel(r'rest-frame wavelength [{\AA}]', fontsize = fsize)
        plt.legend(fontsize = fsize, frameon = True)
        plt.ylim(-1, 10)
        plt.title(r'input: $\log L_{{\rm bol}} = {}\pm{},~\log M_\bullet = {}\pm{}$, output: $\log L_{{\rm bol}} = {}\pm{},~\log M_\bullet = {}\pm{}$'.format(np.round(data[ind_test, 6][0], 2), np.round(1./np.sqrt(data_ivar[ind_test, 6])[0], 2), np.round(data[ind_test, 0][0], 2), np.round(1./np.sqrt(data_ivar[ind_test, 0])[0], 2), np.round(X_new_n[0], 2), np.round(np.sqrt(X_new_var_n[0]), 2), np.round(all_Y_new_i[ind_test][0][0], 2), np.round(np.sqrt(all_Y_new_var_i[ind_test][0][0]), 2)), fontsize = 18)
        fig.savefig('/Users/eilers/Dropbox/projects/RM_black_hole_masses/plots/new/spectra/spectra_prediction_{}_{}.pdf'.format(name, qq), bbox_inches = 'tight')
    

# -------------------------------------------------------------------------------
# sample latent space based on given BH masses
# ------------------------------------------------------------------------------- 

qq = 0
f = open('files/{}_qq{}.pickle'.format(name, qq), 'rb')
all_Y_new_i, all_Y_new_var_i, sort_r, Z_final, Z_opt_n, hyper_params, success_train, success_test = pickle.load(f)
f.close()

theta_band, gamma_band, theta_rbf, gamma_rbf = hyper_params

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

Y_input = data_scaled[ind_train, 0]
Y_var_input = 1./data_ivar_scaled[ind_train, 0]
Y_mask = np.ones_like(Y_input).astype(bool)
Y_mask[np.isnan(Y_input)] = False

X_input = data_scaled[ind_train, 6:]
X_var_input = 1./data_ivar_scaled[ind_train, 6:]
X_mask = np.ones_like(X_input).astype(bool)
X_mask[np.isnan(X_input)] = False

good_stars = Y_mask

zs = np.linspace(-1, 1, 100)
Y_new = np.zeros((len(zs), Q))
#Q_z_grid = np.zeros((len(zs), Q))
for qi in range(Q):
    for zi in range(len(zs)):
        #Q_z_grid[zi, qi] = zs[zi]
        Z_chosen = np.zeros(Q)
        Z_chosen[qi] = zs[zi]
        Y_new_n, Y_new_var_n, k_Z_zj, factor = mean_var(Z_final, Z_chosen, Y_input[good_stars], Y_var_input[good_stars], gamma_rbf, gamma_band)                
        Y_new[zi, qi] = Y_new_n * scales[0] + pivots[0]
        
im = plt.imshow(Y_new, aspect = 'auto')
cbar = plt.colorbar(im)
fig.savefig('/Users/eilers/Dropbox/projects/RM_black_hole_masses/plots/new/Z_MBH_{}_qq{}.pdf'.format(name, qq), bbox_inches = 'tight')

# in principle this should be 16 dimensional...

N_spec = 50
dm = 0.25
logmasses = np.array([6, 7, 8, 9, 10])
spectra_MBH = np.zeros((len(logmasses), N_spec, len(waves)+1))
for i, mi in enumerate(logmasses):
    mask = np.abs(Y_new - logmasses[i]) < dm
    #fig, ax = plt.subplots(1, 1, figsize = (7, 7))
    #ax.imshow(mask, aspect = 'auto')
    all_true = np.transpose(np.nonzero(mask==True))        
    choice = np.random.choice(len(all_true), size = N_spec, replace = True)
    all_true_choice = all_true[choice]
    for ns in range(N_spec):
        Z_chosen_n = np.zeros(Q)
        Z_chosen_n[all_true_choice[ns][1]] = zs[all_true_choice[ns][0]]        
        X_new_n, X_new_var_n, k_Z_zj = mean_var(Z_final, Z_chosen_n, X_input, X_var_input, theta_rbf, theta_band, X_mask)
        X_new_n = X_new_n * scales[6:] + pivots[6:]
        X_new_var_n =  X_new_var_n * scales[6:]**2
        spectra_MBH[i, ns, :] = X_new_n

spectra_MBH_avg = np.mean(spectra_MBH, axis = 1)

inds = np.random.choice(N_spec, size = 10, replace = False)
fig, ax = plt.subplots(2, 1, figsize = (10, 7))
for mi in range(len(logmasses)):
    ax[0].plot(waves, spectra_MBH_avg[mi, 1:], label = r'$\log_{{10}}(M_\bullet/M_\odot) = {}$'.format(int(logmasses[mi])), lw = 2, color = colors[mi])
    ax[1].plot(waves, spectra_MBH_avg[mi, 1:], label = r'$\log_{{10}}(M_\bullet/M_\odot) = {}$'.format(int(logmasses[mi])), lw = 2, color = colors[mi])
    #for i in range(10):
    #    ax[1].plot(waves, spectra_MBH[mi, inds[i], 1:], lw = .8, alpha = .6, color = colors[mi])
ax[0].legend()
ax[1].set_ylim(0, 5)
ax[1].set_xlim(2700, 2900)
fig.savefig('/Users/eilers/Dropbox/projects/RM_black_hole_masses/plots/new/spectraMBH_{}_qq{}.pdf'.format(name, qq), bbox_inches = 'tight')

# -------------------------------------------------------------------------------
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

    