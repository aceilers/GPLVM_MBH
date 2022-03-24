#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 25 16:11:08 2020

@author: eilers
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import astropy.units as u
import astropy.constants as const
from astropy.table import Table, Column
from astropy.convolution import convolve, Box1DKernel
import astropy.cosmology as cosmo
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
from astropy.io import fits
from scipy.stats import gaussian_kde
import scipy.interpolate as interpol

planck = cosmo.Planck13

# -------------------------------------------------------------------------------
# plotting settings
# -------------------------------------------------------------------------------

matplotlib.rc('text', usetex=True)

colors = ["windows blue", "amber", "greyish", "faded green", "dusty purple", "pale red", "dark red"]
colors = sns.xkcd_palette(colors)

lsize = 22
matplotlib.rcParams['ytick.labelsize'] = lsize
matplotlib.rcParams['xtick.labelsize'] = lsize

np.random.seed(42)

# -------------------------------------------------------------------------------
# load data
# -------------------------------------------------------------------------------

t = Table.read('RM_sample.txt', format = 'ascii', header_start = 0, delimiter = '\t')

# rescale black hole masses by different virial factors...


# -------------------------------------------------------------------------------
# plots
# -------------------------------------------------------------------------------

#cuts = t['spectrograph'] != 'SDSS-RM'

fig, ax = plt.subplots(1, 1, figsize = (10, 8))
sc = ax.errorbar(t['sigma_rms_Hbeta'], t['log_MBH_RM'], yerr = t['log_MBH_RM_err'], xerr = t['sigma_rms_Hbeta_err'], fmt = 'o') #color = t['z'], 
#cbar = plt.colorbar(sc)
#cbar.set_label(r'$z$', fontsize = lsize)
ax.tick_params(axis=u'both', direction='in', which='both')
ax.set_xlabel(r'$\sigma_{{\rm H}\beta}$', fontsize = lsize)
ax.set_ylabel(r'$\log M_{\rm BH}$', fontsize = lsize)
plt.savefig('training_set/MBH_Hbeta_2.pdf', bbox_inches = 'tight')

fig, ax = plt.subplots(1, 1, figsize = (10, 8))
sc = ax.scatter(t['sigma_rms_Hbeta'], t['log_MBH_RM'], c = t['z']) 
cbar = plt.colorbar(sc)
cbar.set_label(r'$z$', fontsize = lsize)
ax.tick_params(axis=u'both', direction='in', which='both')
ax.set_xlabel(r'$\sigma_{{\rm H}\beta}$', fontsize = lsize)
ax.set_ylabel(r'$\log M_{\rm BH}$', fontsize = lsize)
plt.savefig('training_set/MBH_Hbeta.pdf', bbox_inches = 'tight')

cuts = t['FWHM_MgII_SE'] > 0
fig, ax = plt.subplots(1, 1, figsize = (10, 8))
sc = ax.scatter(t['FWHM_MgII_SE'][cuts], t['log_MBH_RM'][cuts], c = t['z'][cuts])
cbar = plt.colorbar(sc)
cbar.set_label(r'$z$', fontsize = lsize)
ax.tick_params(axis=u'both', direction='in', which='both')
ax.set_xlabel(r'$\rm FWHM_{\rm MgII}$', fontsize = lsize)
ax.set_ylabel(r'$\log M_{\rm BH}$', fontsize = lsize)
plt.savefig('training_set/MBH_MgII_2.pdf', bbox_inches = 'tight')

cuts = t['FWHM_CIV_SE'] > 0
fig, ax = plt.subplots(1, 1, figsize = (10, 8))
sc = ax.scatter(t['sigma_CIV_SE'][cuts], t['log_MBH_RM'][cuts], c = t['z'][cuts])
cbar = plt.colorbar(sc)
cbar.set_label(r'$z$', fontsize = lsize)
ax.tick_params(axis=u'both', direction='in', which='both')
ax.set_xlabel(r'$\sigma_{\rm CIV}$', fontsize = lsize)
ax.set_ylabel(r'$\log M_{\rm BH}$', fontsize = lsize)
plt.savefig('training_set/MBH_CIV.pdf', bbox_inches = 'tight')

cuts = t['log_Lbol_3000'] > 0
fig, ax = plt.subplots(1, 1, figsize = (10, 8))
sc = ax.scatter(t['log_Lbol_3000'][cuts], t['log_MBH_RM'][cuts], c = t['z'][cuts])
cbar = plt.colorbar(sc)
cbar.set_label(r'$z$', fontsize = lsize)
ax.tick_params(axis=u'both', direction='in', which='both')
ax.set_xlabel(r'$\log L_{\rm bol, 3000A}$', fontsize = lsize)
ax.set_ylabel(r'$\log M_{\rm BH}$', fontsize = lsize)
plt.savefig('training_set/MBH_logLbol_2.pdf', bbox_inches = 'tight')

cuts = t['log_Lbol_1350'] > 0
fig, ax = plt.subplots(1, 1, figsize = (10, 8))
sc = ax.scatter(t['log_Lbol_1350'][cuts], t['log_MBH_RM'][cuts], c = t['z'][cuts])
cbar = plt.colorbar(sc)
cbar.set_label(r'$z$', fontsize = lsize)
ax.tick_params(axis=u'both', direction='in', which='both')
ax.set_xlabel(r'$\log L_{\rm bol, 1350A}$', fontsize = lsize)
ax.set_ylabel(r'$\log M_{\rm BH}$', fontsize = lsize)
plt.savefig('training_set/MBH_logLbol.pdf', bbox_inches = 'tight')


# Park et al. 2017
cuts = t['log_Lbol_1350'] > 0
MBH_SE = 6.9 + 0.44 * np.log10(10**t['log_Lbol_1350'][cuts]/1e44) + 1.66 * np.log10(t['sigma_CIV_SE'][cuts]/1000)
xx = np.linspace(6, 10, 100)
fig, ax = plt.subplots(1, 1, figsize = (10, 8))
sc = ax.scatter(MBH_SE, t['log_MBH_RM'][cuts], c = t['z'][cuts])
cbar = plt.colorbar(sc)
cbar.set_label(r'$z$', fontsize = lsize)
ax.plot(xx, xx, linestyle ='--', color = colors[2])
ax.set_xlim(6.4, 9.2)
ax.set_ylim(6.4, 9.2)
ax.tick_params(axis=u'both', direction='in', which='both')
ax.set_xlabel(r'$\log M_{\rm BH}$~(SE)', fontsize = lsize)
ax.set_ylabel(r'$\log M_{\rm BH}$~(RM)', fontsize = lsize)
plt.savefig('training_set/MBH_Park2017.pdf', bbox_inches = 'tight')

# Bhak et al. 2019
cuts = t['log_Lbol_3000'] > 0
MBH_SE = 7.13 + 0.49 * np.log10(10**t['log_Lbol_3000'][cuts]/1e44) + 2.7 * np.log10(t['sigma_MgII_SE'][cuts]/1000)
xx = np.linspace(6, 10, 100)
fig, ax = plt.subplots(1, 1, figsize = (10, 8))
sc = ax.scatter(MBH_SE, t['log_MBH_RM'][cuts], c = t['z'][cuts])
cbar = plt.colorbar(sc)
cbar.set_label(r'$z$', fontsize = lsize)
ax.plot(xx, xx, linestyle ='--', color = colors[2])
ax.set_xlim(6.1, 9.2)
ax.set_ylim(6.1, 9.2)
ax.tick_params(axis=u'both', direction='in', which='both')
ax.set_xlabel(r'$\log M_{\rm BH}$~(SE)', fontsize = lsize)
ax.set_ylabel(r'$\log M_{\rm BH}$~(RM)', fontsize = lsize)
plt.savefig('training_set/MBH_Bhak2019.pdf', bbox_inches = 'tight')

fig, ax = plt.subplots(1, 1, figsize = (8, 8))
sc = ax.errorbar(MBH_SE, t['log_MBH_RM'][cuts], yerr = t['log_MBH_RM_err'][cuts], fmt = 'o', capsize = 2, capthick = 2)
ax.plot(xx, xx, linestyle ='--', color = colors[2])
ax.set_xlim(6, 9.5)
ax.set_ylim(6, 9.5)
ax.tick_params(axis=u'both', direction='in', which='both')
ax.set_xlabel(r'$\log M_{\rm BH}$~(SE)', fontsize = lsize)
ax.set_ylabel(r'$\log M_{\rm BH}$~(RM)', fontsize = lsize)
plt.savefig('training_set/MBH_Bhak2019_2.pdf', bbox_inches = 'tight')






