#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 12 15:34:41 2021

@author: eilers
"""
import numpy as np
from astropy import units as u
import astropy.cosmology as cosmo
from astropy import constants as const
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from scipy.stats import binned_statistic_2d
from scipy.ndimage import gaussian_filter
import matplotlib.gridspec as gridspec
from astropy.table import Column, Table, join, vstack, hstack, unique
from astropy.convolution import convolve, Box1DKernel, Gaussian1DKernel
import pickle
import emcee
import corner
from astropy.io import fits
from scipy.optimize import minimize
import scipy.optimize as op
from astropy.io import ascii
from _cont_src import fit_sdss_cont
from astropy.stats import sigma_clip

from stacking import compute_stack

planck = cosmo.Planck18

# -------------------------------------------------------------------------------
# plotting settings
# -------------------------------------------------------------------------------

colors = ["windows blue", "amber", "greyish", "faded green", "dusty purple", "pale red", "orange", "red", "blue", "lime green"]
colors = sns.xkcd_palette(colors)
#colors = sns.color_palette("Blues")

matplotlib.rcParams['ytick.labelsize'] = 24
matplotlib.rcParams['xtick.labelsize'] = 24
matplotlib.rc('text', usetex=True)
fsize = 24

colors_cont = ["black", "grey", "light grey"] 
colors_cont = sns.xkcd_palette(colors_cont)

np.random.seed(40)

# -------------------------------------------------------------------------------
# To Do
# -------------------------------------------------------------------------------
'''
1. pick higher SN spectrum in case there are two
2. when chosing unique rows, decide which value to take
3. what to do with quasars that don't have L3000 nor L5100? using bolometric correction for L1450 now for L1700...
'''

# -------------------------------------------------------------------------------
# Shen+2019 all SDSS-RM targets
# -------------------------------------------------------------------------------

S19 = Table.read('../RM_black_hole_masses/Shen2019_SDSSRM.fits', format = 'fits')

#cuts = (S19['LOGBH_MGII_S11'] > 0) * (S19['LOGLBOL'] > 0)
#plt.scatter(S19['LOGBH_MGII_S11'][cuts], S19['LOGLBOL'][cuts])
#plt.hist(S19['LOGBH_MGII_S11'][cuts], bins = 30)

# -------------------------------------------------------------------------------
# Grier+2019: 48 CIV lag RM BH measurements
# -------------------------------------------------------------------------------

# #G19_1 = Table.read('Grier2019_table1.txt', format = 'ascii.cds')
# G19_4 = Table.read('Grier2019_table4_mod.txt', format = 'ascii') 
# G19_4['logMBH'] = np.log10(G19_4['MBH'] * 1e7)
# G19_4['E_logMBH'] = G19_4['MBH_err_p'] / (G19_4['MBH'] * np.log(10))
# G19_4['e_logMBH'] = G19_4['MBH_err_m'] / (G19_4['MBH'] * np.log(10))
# G19_4['survey'] = 'G19_SDSSRM_CIV'

# # no overlap between Halpha/Hbeta and CIV targets, so it's difficult to estimate scatter between the samples

# -------------------------------------------------------------------------------
# Grier+2017: 44 Hbeta lag RM BH measurements + 18 Halpha RM BH measurements (partially overlapping!)
# -------------------------------------------------------------------------------

G17_4 = Table.read('../RM_black_hole_masses/Grier2017_table4_Hbeta.txt', format = 'ascii') 
G17_4['logMBH'] = np.log10(G17_4['MBH'] * 1e7)
G17_4['E_logMBH'] = G17_4['MBH_err_p'] / (G17_4['MBH'] * np.log(10))
G17_4['e_logMBH'] = G17_4['MBH_err_m'] / (G17_4['MBH'] * np.log(10))
G17_4['survey'] = 'G17_SDSSRM_Hbeta'

G17_5 = Table.read('../RM_black_hole_masses/Grier2017_table5_Halpha.txt', format = 'ascii', delimiter = '\t')
G17_5['logMBH'] = np.log10(G17_5['MBH'] * 1e7)
G17_5['E_logMBH'] = G17_5['MBH_err_p'] / (G17_5['MBH'] * np.log(10))
G17_5['e_logMBH'] = G17_5['MBH_err_m'] / (G17_5['MBH'] * np.log(10))
G17_5['survey'] = 'G17_SDSSRM_Halpha'

# assess scatter between Halpha and Hbeta
both_samples = [x for x in G17_5['RMID'] if x in G17_4['RMID']]

mask_ha_only = np.ones(len(G17_5), dtype = bool)
mbh_hbeta = np.zeros(len(both_samples))
mbh_halpha = np.zeros(len(both_samples))
for i in range(len(both_samples)):
    mbh_hbeta[i] = G17_4['logMBH'][G17_4['RMID'] == both_samples[i]]
    mbh_halpha[i] = G17_5['logMBH'][G17_5['RMID'] == both_samples[i]]
    mask_ha_only[list(G17_5['RMID']).index(both_samples[i])] = False
offset_halpha = np.percentile(mbh_hbeta - mbh_halpha, 50)
scatter_halpha = 0.5*(np.percentile(mbh_hbeta - mbh_halpha, 84) - np.percentile(mbh_hbeta - mbh_halpha, 16))

# mask one object with huge uncertainties
mask_ha_only[G17_5['RMID'] == 88] = False
#mask_ha_only[G17_5['RMID'] == 252] = False # no emission line visible in spectrum...

# take Hbeta measurements where possible
G17_5 = G17_5[mask_ha_only]

# correct measurement offset! 
# G17_5['logMBH'] += offset_halpha

# -------------------------------------------------------------------------------
# Homayouni+2020: 57 MgII RM measurements
# -------------------------------------------------------------------------------

# H20 = Table.read('Homayouni2020.txt', format = 'ascii.cds')
# H20['survey'] = 'H20_SDSSRM_MgII'
# selection = H20['Gold'] == 1
# H20 = H20[selection]

# both_samples = [x for x in H20['RMID'] if x in G17_4['RMID']]
# # large discrepancies between H20 results and G17 for 4 overlapping targets!

# -------------------------------------------------------------------------------
# merge tables
# -------------------------------------------------------------------------------

xx = vstack([G17_4, G17_5]) # vstack([G19_4, G17_4, G17_5, H20])
xx.sort('RMID')

xx_new = join(S19, xx, keys = 'RMID')

# -------------------------------------------------------------------------------
# download spectra
# -------------------------------------------------------------------------------

xx = xx_new.copy()

found = np.ones((len(xx)), dtype = bool)
f = open('../RM_black_hole_masses/BOSS/download_spectra.txt', 'w') 
for i in range(len(xx)):
    if len(xx['PLATE_ALL'][i].split()) == 1:
        f.write('{},{},{} \n'.format(xx['PLATE_ALL'][i].split()[0], xx['MJD_ALL'][i].split()[0], xx['FIBERID_ALL'][i].split()[0]))
    elif len(xx['PLATE_ALL'][i].split()) == 2:
        f.write('{},{},{} \n'.format(xx['PLATE_ALL'][i].split()[0], xx['MJD_ALL'][i].split()[0], xx['FIBERID_ALL'][i].split()[0]))
        f.write('{},{},{} \n'.format(xx['PLATE_ALL'][i].split()[1], xx['MJD_ALL'][i].split()[1], xx['FIBERID_ALL'][i].split()[1]))
    elif len(xx['PLATE_ALL'][i].split()) == 0:
        found[i] = False
f.close()    

xx = xx[found]  # 138 entries with spectra!

# -------------------------------------------------------------------------------
# applied virial factor
# -------------------------------------------------------------------------------

log_f = 0.65
log_f_err = 0.12

# -------------------------------------------------------------------------------
# look at spectra
# -------------------------------------------------------------------------------

# zz = fits.open('BOSS/spec-7031-56449-0512.fits')
# data = zz[1].data
# header = zz[0].header


def PowerLaw(F0, alpha, wl):    
    return F0 * (wl/2500.)**alpha

def lnlike(pars, flux, ivar, wl):
    F0, alpha = pars
    return np.sum((flux - PowerLaw(F0, alpha, wl))**2 * ivar)

# -------------------------------------------------------------------------------
# wavelength grid
# -------------------------------------------------------------------------------

# needs to include: MBH, z, SNR, Lbol, quality flag, spectra

# hbeta = 4861
# halpha = 6563

min_waves, max_waves = 1220, 5000.1
wave_grid = np.arange(min_waves, max_waves, 2)

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

dline = 60.
dlam_noline = 20.
dlam_line = 2.

lines = np.array([lya, siiv, civ, ciii, mgii, hbeta])


# wave_grid = list([min_waves]) #np.arange(max(np.round(lya) - dline, min_waves), np.round(lya) + dline, 2))
# for l in lines:
#     if wave_grid[-1] > l-dline and wave_grid[-1] < l+dline and wave_grid[-1] < max_waves:
#         while wave_grid[-1] > l-dline and wave_grid[-1] < l+dline:
#             wave_grid.append(wave_grid[-1] + dlam_line)
#     elif wave_grid[-1] < l-dline:
#         while wave_grid[-1] < l-dline:
#             wave_grid.append(wave_grid[-1] + dlam_noline)
#         if wave_grid[-1] > l-dline and wave_grid[-1] < l+dline:
#             while wave_grid[-1] > l-dline and wave_grid[-1] < l+dline and wave_grid[-1] < 5000.1:
#                 wave_grid.append(wave_grid[-1] + dlam_line)
# if wave_grid[-1] < max_waves:
#     while wave_grid[-1] < max_waves:          
#         wave_grid.append(wave_grid[-1] + dlam_noline)

wave_grid = np.array(wave_grid)
print('number of pixels: {}'.format(len(wave_grid)))


# -------------------------------------------------------------------------------
# create rectangular data
# -------------------------------------------------------------------------------

# masked low-redshift objects
masked_lowz = xx['ZSYS'] > (np.around(4000/max_waves - 1, decimals = 1) + 0.1)
xx = xx[masked_lowz]

data = np.zeros((len(xx), len(wave_grid)+5)) 
data_ivar = np.zeros((len(xx), len(wave_grid)+5)) 

dl = 5

# table with BOSS spectra
boss = Table.read('../RM_black_hole_masses/BOSS/optical_search_287819.csv')

for i in range(len(xx)):

    if len(xx['PLATE_ALL'][i].split()) == 1:
        zz = fits.open('../RM_black_hole_masses/BOSS/spec-{}-{}-{}.fits'.format(xx['PLATE_ALL'][i].split()[0], xx['MJD_ALL'][i].split()[0], xx['FIBERID_ALL'][i].split()[0]))
        find = (boss['#plate'] == int(xx['PLATE_ALL'][i].split()[0])) * (boss['mjd'] == int(xx['MJD_ALL'][i].split()[0])) * (boss['fiberid'] == int(xx['FIBERID_ALL'][i].split()[0]))
        snr = float(boss['sn_median_r'][find])  
   
    elif len(xx['PLATE_ALL'][i].split()) == 2:
        find1 = (boss['#plate'] == int(xx['PLATE_ALL'][i].split()[0])) * (boss['mjd'] == int(xx['MJD_ALL'][i].split()[0])) * (boss['fiberid'] == int(xx['FIBERID_ALL'][i].split()[0]))
        find2 = (boss['#plate'] == int(xx['PLATE_ALL'][i].split()[1])) * (boss['mjd'] == int(xx['MJD_ALL'][i].split()[1])) * (boss['fiberid'] == int(xx['FIBERID_ALL'][i].split()[1]))
        snr1 = float(boss['sn_median_r'][find1])
        snr2 = float(boss['sn_median_r'][find2])
        #print(i, snr1, xx['PLATE_ALL'][i].split()[0], xx['MJD_ALL'][i].split()[0], xx['FIBERID_ALL'][i].split()[0])
        #print(i, snr2, xx['PLATE_ALL'][i].split()[1], xx['MJD_ALL'][i].split()[1], xx['FIBERID_ALL'][i].split()[1])
        if snr1 >= snr2:
            zz = fits.open('../RM_black_hole_masses/BOSS/spec-{}-{}-{}.fits'.format(xx['PLATE_ALL'][i].split()[0], xx['MJD_ALL'][i].split()[0], xx['FIBERID_ALL'][i].split()[0]))
            snr = snr1
        elif snr2 > snr1:
            zz = fits.open('../RM_black_hole_masses/BOSS/spec-{}-{}-{}.fits'.format(xx['PLATE_ALL'][i].split()[1], xx['MJD_ALL'][i].split()[1], xx['FIBERID_ALL'][i].split()[1]))
            snr = snr2
    
    data_q = zz[1].data
    header = zz[0].header
    wave = 10**(header['COEFF0'] + header['COEFF1'] * np.arange(len(data_q['flux'])))
    
    #remove edge effetcs:
    cuts = (wave > 4000) * (wave < 10000)
    wave = wave[cuts]
    flux = data_q['flux'][cuts]
    ivar = data_q['ivar'][cuts]
    wave_rest = wave / (1+xx['ZSYS'][i])

    # remove strange absorption feature:
    if xx['RMID'][i] == 373:
        masked = (wave_rest < 2782) * (wave_rest > 2765)
        flux = flux[~masked]
        ivar = ivar[~masked]
        wave_rest = wave_rest[~masked]    
    
    # mask absoprtion lines
    qsos_cont = np.zeros_like(flux)
    nfit = len(flux)
    if xx['RMID'][i] == 17 or xx['RMID'][i] == 101: 
        deltapix2 = 5
    elif xx['RMID'][i] == 16 or xx['RMID'][i] == 21:
        deltapix2 = 30
    elif xx['RMID'][i] == 33 or xx['RMID'][i] == 457:
        deltapix = 20
    else:
        deltapix2 = 15
    fit_sdss_cont(wave_rest, flux, 1./np.sqrt(ivar), nfit, 0, 20, deltapix2, 4, 0.033, 0.99, qsos_cont, 1.0)
    clipped = sigma_clip((flux-qsos_cont), sigma_lower = 3, sigma_upper = 4) 
    # plt.plot(wave_rest, flux, color = '0.8')
    # plt.plot(wave_rest[~clipped.mask], flux[~clipped.mask], color = 'k', label = 'RMID: {}, dpix={}'.format(xx['RMID'][i], deltapix2))
    # plt.plot(wave_rest, qsos_cont, color = 'r', zorder = 10)
    # plt.xlim(1220, 2000)
    # plt.ylim(0, 30)
    # plt.legend()
    # plt.show()
    flux_orig = flux.copy()
    wave_orig = wave_rest.copy()
    flux = flux[~clipped.mask]
    ivar = ivar[~clipped.mask]
    wave_rest = wave_rest[~clipped.mask]
    
    # # mask emission lines
    # inds = (wave_rest > min_waves) * (wave_rest < max_waves)
    # yy = convolve(flux[inds], Box1DKernel(500))
    # clipped = sigma_clip((flux[inds]-yy)*np.sqrt(ivar[inds]), sigma_lower = 1, sigma_upper = 1)
    # # fig = plt.subplots(1, 1, figsize = (12, 6))
    # # plt.plot(wave_rest[inds], flux[inds])
    # # #plt.plot(qsos[q].wave_rest, yy)
    # # plt.plot(wave_rest[inds][~clipped.mask], flux[inds][~clipped.mask])
    # mean = np.nanmean(flux[inds][~clipped.mask])
    # print(i, mean)
    # flux_norm = flux / mean
    # ivar_norm = ivar * mean**2
    # # plt.axhline(mean)
    # # plt.show()    
    
    # -------------------------------------------------------------------------------
    # fit power-law
    # -------------------------------------------------------------------------------

    # areas free of emission lines
    good_indices = np.zeros_like(wave_rest, dtype = bool)
    good_indices[(wave_rest > 1270) * (wave_rest < 1280)] = True
    good_indices[(wave_rest > 1350) * (wave_rest < 1370)] = True
    good_indices[(wave_rest > 1430) * (wave_rest < 1470)] = True
    good_indices[(wave_rest > 1600) * (wave_rest < 1620)] = True
    good_indices[(wave_rest > 1690) * (wave_rest < 1730)] = True
    good_indices[(wave_rest > 1770) * (wave_rest < 1850)] = True
    good_indices[(wave_rest > 1950) * (wave_rest < 2700)] = True
    good_indices[(wave_rest > 2900) * (wave_rest < 3400)] = True
    good_indices[(wave_rest > 3450) * (wave_rest < 3700)] = True
    good_indices[(wave_rest > 3750) * (wave_rest < 3850)] = True
    good_indices[(wave_rest > 3900) * (wave_rest < 4300)] = True
    good_indices[(wave_rest > 4400) * (wave_rest < 4800)] = True
    
    x0 = np.array([1, -1.2])
    bnds = [(0, None), (-2.5, 0)]
    res = minimize(lnlike, x0 = x0, args = (flux[good_indices], ivar[good_indices], wave_rest[good_indices]), bounds = bnds)
    F0, alpha = res.x
    # normalization 
    mean = PowerLaw(F0, alpha, 2500)
    print(i, mean)
    flux_norm = flux / mean
    ivar_norm = ivar * mean**2

    # -------------------------------------------------------------------------------
    # interpolate to common wavelength grid
    # -------------------------------------------------------------------------------
   
    # interpolate to new wavelengths
    wave_stack, flux_stack, ivar_stack, mask_stack, nused = compute_stack(wave_grid, wave_rest, flux_norm, ivar_norm, 
                                                                                                  np.ones_like(wave_rest, dtype=bool), np.ones_like(wave_rest))    

    # missing data   
    flux_stack[flux_stack == 0] = np.nan
    ivar_stack[np.isinf(ivar_stack)] = np.nan
    
    fig = plt.figure(figsize = (12, 6))
    plt.plot(wave_orig, flux_orig / mean, color = '0.9', label = r'original data: RMID {}, $z={}$, SNR$={}$, $\log M_\bullet = {}$'.format(xx['RMID'][i], np.round(xx['ZSYS'][i], 3), snr, np.round(xx['logMBH'][i], 2)))
    plt.plot(wave_orig, qsos_cont / mean, color = 'r', zorder = 10, label = 'spline fit', lw = 2)
    plt.plot(wave_stack, flux_stack, color = 'k', label = 'input data', zorder  = 12)
    plt.plot(wave_stack, 1/np.sqrt(ivar_stack), color = '0.7', label = 'input noise')
    plt.plot(wave_stack, PowerLaw(F0, alpha, wave_stack) / mean, label = r'power law, $\alpha={}$'.format(round(alpha, 2)), lw = 1, linestyle = '--', color = 'blue')
    plt.ylim(-1, 6)
    plt.xlim(1190, 5000.1)
    #plt.axhline(mean, color = colors[5], lw = 2)
    plt.legend(fontsize = 16)
    plt.savefig('../RM_black_hole_masses/BOSS/plots/spectrum_SDSSRM_{}.pdf'.format(i))
    plt.close()
    

    data[i, 6:] = flux_stack
    data_ivar[i, 6:] = ivar_stack
    
    data[i, 0] = xx['logMBH'][i]
    if xx['survey'][i] == 'G17_SDSSRM_Hbeta':
        add_error = 0
    elif xx['survey'][i] == 'G17_SDSSRM_Halpha':
        add_error = scatter_halpha
    # elif xx['survey'][i] == 'G19_SDSSRM_CIV':
    #     add_error = 0
    # elif xx['survey'][i] == 'H20_SDSSRM_MgII':
    #     add_error = 0
    data_ivar[i, 0] = 1 / ((0.5*(xx['E_logMBH'][i] + xx['e_logMBH'][i]))**2 + log_f_err**2 + add_error**2) # uncertainty from f
    data[i, 1] = xx['ZSYS'][i]
    data_ivar[i, 1] = 1 / xx['ZSYS_ERR'][i]**2
    data[i, 2] = snr
    data_ivar[i, 2] = 0
    data[i, 3] = alpha
    data_ivar[i, 3] = np.nan
    # if xx['survey'][i] == 'G17_SDSSRM_Hbeta':
    #     data[i, 4] = 1
    # elif xx['survey'][i] == 'G17_SDSSRM_Halpha':
    #     data[i, 4] = 2
    # elif xx['survey'][i] == 'G19_SDSSRM_CIV':
    #     data[i, 4] = 3
    # elif xx['survey'][i] == 'H20_SDSSRM_MgII':
    #     data[i, 4] = 4
    # data_ivar[i, 4] = np.nan
    data[i, 4] = xx['lag'][i]
    data_ivar[i, 4] = (xx['lag_err'][i])**(-2)
    data[i, 5] = mean
    data_ivar[i, 5] = np.nan
    
    #print(i, xx['LOGL1700'][i], xx['LOGL3000'][i], xx['LOGL5100'][i])
    if xx['LOGL3000'][i] > 0:
        # Richards+ 2006
        log_Lbol = np.log10(5.15) + xx['LOGL3000'][i]
        log_Lbol_err = xx['LOGL3000_ERR'][i] 
        #Lbol = 0.75 * (1.85 + 0.97 * xx['LOGL3000'][i])
        #Lbol_err = 0.75 * np.sqrt(1.27**2 + (xx['LOGL3000'][i] * 0.03)**2 + (0.97 * xx['LOGL3000_ERR'][i])**2)
        data[i, 6] = log_Lbol
        data_ivar[i, 6] = 1 / log_Lbol_err**2
    elif xx['LOGL5100'][i] > 0:
        #Lbol = 0.75 * (4.89 + 0.91 * xx['LOGL5100'][i])
        #Lbol_err = 0.75 * np.sqrt(1.66**2 + (xx['LOGL5100'][i] * 0.04)**2 + (0.91 * xx['LOGL5100_ERR'][i])**2)
        log_Lbol = np.log10(9.26) + xx['LOGL5100'][i]
        log_Lbol_err = xx['LOGL5100_ERR'][i]         
        data[i, 6] = log_Lbol
        data_ivar[i, 6] = 1 / log_Lbol_err**2
    elif xx['LOGL1700'][i] > 0:
        #Lbol = 0.75 * (4.74 + 0.91 * xx['LOGL1700'][i])
        #Lbol_err = 0.75 * np.sqrt(1.**2 + (xx['LOGL1700'][i] * 0.02)**2 + (0.91 * xx['LOGL1700_ERR'][i])**2)
        log_Lbol = np.log10(3.81) + xx['LOGL1700'][i] # THIS IS FOR 1350!! Close enough, see Richards 2006b
        log_Lbol_err = xx['LOGL1700_ERR'][i]
        data[i, 6] = log_Lbol
        data_ivar[i, 6] = 1 / log_Lbol_err**2
    else:
        print(i)                
        
 
print('done!')

# -------------------------------------------------------------------------------
# addding HST data
# -------------------------------------------------------------------------------

# HST data
f = open('../RM_black_hole_masses/data_HST_1220_5000_2A.pickle', 'rb')
data_hst, data_ivar_hst = pickle.load(f)
f.close()

data_all = np.vstack([data, data_hst])
data_ivar_all = np.vstack([data_ivar, data_ivar_hst])

f = open('../RM_black_hole_masses/data_HST_SDSS_1220_5000_2A.pickle', 'wb')
pickle.dump([data_all, data_ivar_all], f)
f.close()

# -------------------------------------------------------------------------------
# parent sample plot
# -------------------------------------------------------------------------------

fig, ax = plt.subplots(1, 2, figsize = (16, 7), sharey = True)
plt.subplots_adjust(wspace = 0.05)
ha = xx['survey'] == 'G17_SDSSRM_Halpha'
ax[0].errorbar(data[ha, 1], data[ha, 0], yerr = 1/np.sqrt(data_ivar[ha, 0]), fmt = 'o', color = colors[1], label = r'SDSS-RM: H$\alpha$ (${}$)'.format(len(data[ha, 0])))
hb = xx['survey'] == 'G17_SDSSRM_Hbeta'
ax[0].errorbar(data[hb, 1], data[hb, 0], yerr = 1/np.sqrt(data_ivar[hb, 0]), fmt = 'o', color = colors[0], label = r'SDSS-RM: H$\beta$ (${}$)'.format(len(data[hb, 0])))
#hb = xx['survey'] == 'G19_SDSSRM_CIV'
#plt.errorbar(xx['ZSYS'][hb], xx['logMBH'][hb], yerr = [xx['E_logMBH'][hb], xx['e_logMBH'][hb]], fmt = 'o', color = colors[2], label = r'SDSS-RM: C\,{{\small IV}} (${}$)'.format(len(xx['ZSYS'][hb])))
#hb = xx['survey'] == 'H20_SDSSRM_MgII'
#plt.errorbar(xx['ZSYS'][hb], xx['logMBH'][hb], yerr = [xx['E_logMBH'][hb], xx['e_logMBH'][hb]], fmt = 'o', color = colors[3], label = r'SDSS-RM: Mg\,{{\small II}} (${}$)'.format(len(xx['ZSYS'][hb])))
ax[0].errorbar(data_hst[:, 1], data_hst[:, 0], yerr = 1/np.sqrt(data_ivar_hst[:, 0]), fmt = 'o', color = colors[5], label = r'HST: H$\beta$ (${}$)'.format(len(data_hst[:, 1])))
ax[0].set_xlabel(r'$z$', fontsize = fsize)
ax[0].set_ylabel(r'$\log_{10}(M_\bullet/M_\odot)$', fontsize = fsize)


ax[1].errorbar(data[hb, 6], data[hb, 0], yerr = 1/np.sqrt(data_ivar[hb, 0]), xerr = 1/np.sqrt(data_ivar[hb, 6]), color = colors[0], fmt = 'o', label = r'SDSS-RM: H$\beta$ (${}$)'.format(len(data[hb, 0])))
ax[1].errorbar(data[ha, 6], data[ha, 0], yerr = 1/np.sqrt(data_ivar[ha, 0]), xerr = 1/np.sqrt(data_ivar[ha, 6]), color = colors[1], fmt = 'o', label = r'SDSS-RM: H$\alpha$ (${}$)'.format(len(data[ha, 0])))
#plt.errorbar(data_all[c, 6], data_all[c, 0], yerr = 1/np.sqrt(data_ivar_all[c, 0]), xerr = 1/np.sqrt(data_ivar_all[c, 6]), color = colors[2], fmt = 'o', label = r'SDSS-RM: C{\small IV}')
#plt.errorbar(data_all[m, 6], data_all[m, 0], yerr = 1/np.sqrt(data_ivar_all[m, 0]), xerr = 1/np.sqrt(data_ivar_all[m, 6]), color = colors[3], fmt = 'o', label = r'SDSS-RM: Mg{\small II}')
ax[1].errorbar(data_hst[:, 6], data_hst[:, 0], yerr = 1/np.sqrt(data_ivar_hst[:, 0]), xerr = 1/np.sqrt(data_ivar_hst[:, 6]), color = colors[5], fmt = 'o', label = r'HST: H$\beta$ (${}$)'.format(len(data_hst[:, 1])))
#plt.errorbar(data_all[cuts_highz, 6], data_all[cuts_highz, 0], yerr = 1/np.sqrt(data_ivar_all[cuts_highz, 0]), xerr = 1/np.sqrt(data_ivar_all[cuts_highz, 6]), color= colors[4], fmt = 'o', label = r'X-Shooter: $z>6$')
#plt.ylabel(r'$\log_{10}\,(M_\bullet/M_\odot)$', fontsize = fsize)
ax[1].set_xlabel(r'$\log_{10}\,(L_{\rm bol} / \rm erg\,s^{-1})$', fontsize = fsize)
ax[0].tick_params(axis=u'both', direction='in', which='both')
ax[1].tick_params(axis=u'both', direction='in', which='both')
ax[1].legend(fontsize = 17, frameon = True, loc = 2)

plt.savefig('../RM_black_hole_masses/plots/rm_training_sample.pdf', bbox_inches = 'tight')

# -------------------------------------------------------------------------------
# corner plot
# -------------------------------------------------------------------------------

#cuts_highz = (data_all[:, 1] > 5) #* (data_all[:, 6] < 47)
# hb = data_all[:, 4] == 1
# ha = data_all[:, 4] == 2
# c = data_all[:, 4] == 3
# m = data_all[:, 4] == 4
# h = data_all[:, 4] == 5
hb = xx['survey'] == 'G17_SDSSRM_Halpha'
fig = plt.subplots(1, 1, figsize = (6, 6))
plt.errorbar(data[hb, 6], data[hb, 0], yerr = 1/np.sqrt(data_ivar[hb, 0]), xerr = 1/np.sqrt(data_ivar[hb, 6]), color = colors[0], fmt = 'o', label = r'SDSS-RM: H$\beta$')
plt.errorbar(data[ha, 6], data[ha, 0], yerr = 1/np.sqrt(data_ivar[ha, 0]), xerr = 1/np.sqrt(data_ivar[ha, 6]), color = colors[1], fmt = 'o', label = r'SDSS-RM: H$\alpha$')
#plt.errorbar(data_all[c, 6], data_all[c, 0], yerr = 1/np.sqrt(data_ivar_all[c, 0]), xerr = 1/np.sqrt(data_ivar_all[c, 6]), color = colors[2], fmt = 'o', label = r'SDSS-RM: C{\small IV}')
#plt.errorbar(data_all[m, 6], data_all[m, 0], yerr = 1/np.sqrt(data_ivar_all[m, 0]), xerr = 1/np.sqrt(data_ivar_all[m, 6]), color = colors[3], fmt = 'o', label = r'SDSS-RM: Mg{\small II}')
plt.errorbar(data_hst[:, 6], data_hst[:, 0], yerr = 1/np.sqrt(data_ivar_hst[:, 0]), xerr = 1/np.sqrt(data_ivar_hst[:, 6]), color = colors[5], fmt = 'o', label = r'HST: H$\beta$')
#plt.errorbar(data_all[cuts_highz, 6], data_all[cuts_highz, 0], yerr = 1/np.sqrt(data_ivar_all[cuts_highz, 0]), xerr = 1/np.sqrt(data_ivar_all[cuts_highz, 6]), color= colors[4], fmt = 'o', label = r'X-Shooter: $z>6$')
plt.ylabel(r'$\log_{10}\,(M_\bullet/M_\odot)$', fontsize = fsize)
plt.xlabel(r'$\log_{10}\,(L_{\rm bol} / \rm erg\,s^{-1})$', fontsize = fsize)
plt.legend(fontsize = 16, frameon = True)
plt.savefig('../RM_black_hole_masses/plots/MBHvsLbol.pdf', bbox_inches = 'tight')
plt.close()

plt.hist(data_all[:, 6])
plt.xlabel(r'$\log_{10}\,(L_{\rm bol} / \rm erg\,s^{-1})$', fontsize = fsize)
plt.savefig('../RM_black_hole_masses/plots/Lbol_hist.pdf', bbox_inches = 'tight')
plt.close()

plt.hist(data_all[:, 0])
plt.xlabel(r'$\log_{10}\,(M_\bullet/M_\odot)$', fontsize = fsize)
plt.savefig('../RM_black_hole_masses/plots/MBH_hist.pdf', bbox_inches = 'tight')
plt.close()

yy = np.nanmean(data_all, axis = 0)
fig = plt.subplots(1, 1, figsize = (14, 6))
plt.plot(wave_grid[1:-1], yy[7:], color = 'k')
plt.xlabel('rest-frame wavelength', fontsize = fsize)
plt.ylabel('flux', fontsize = fsize)
plt.savefig('../RM_black_hole_masses/plots/composite_spectrum_input_data.pdf', bbox_inches = 'tight')


# f = open('data_HST_SDSS_bin2_5000.pickle', 'rb')
# data, data_ivar = pickle.load(f)
# f.close()

# min_waves, max_waves = 1220, 5000.1
# wave_grid = np.arange(min_waves, max_waves, 2)

# i1, i2, i3 = 1, 44, 72
# fig, ax = plt.subplots(3, 1, figsize = (12, 8), sharex = True)
# ax[0].plot(wave_grid[1:-1], data[i1, 7:], color = 'k', label = r'$z={}$, SNR$={}$, $\log M_\bullet = {}$'.format(np.round(data_all[i1, 1], 3), np.round(data_all[i1, 2], 3), np.round(data_all[i1, 0], 2)))
# ax[0].plot(wave_grid[1:-1], 1/np.sqrt(data_ivar[i1, 7:]), color = '0.7')
# ax[1].plot(wave_grid[1:-1], data[i2, 7:], color = 'k', label = r'$z={}$, SNR$={}$, $\log M_\bullet = {}$'.format(np.round(data_all[i2, 1], 3), np.round(data_all[i2, 2], 3), np.round(data_all[i2, 0], 2)))
# ax[1].plot(wave_grid[1:-1], 1/np.sqrt(data_ivar[i2, 7:]), color = '0.7')
# ax[2].plot(wave_grid[1:-1], data[i3, 7:], color = 'k', label = r'$z={}$, SNR$={}$, $\log M_\bullet = {}$'.format(np.round(data_all[i3, 1], 3), np.round(data_all[i3, 2], 3), np.round(data_all[i3, 0], 2)))
# ax[2].plot(wave_grid[1:-1], 1/np.sqrt(data_ivar[i3, 7:]), color = '0.7')

# ax[0].set_ylim(-1, 6)
# ax[0].set_xlim(1220, 5000.1)
# #plt.axhline(mean, color = colors[5], lw = 2)
# ax[0].legend(fontsize = 16)
# ax[1].legend(fontsize = 16)
# ax[2].legend(fontsize = 16)
# plt.subplots_adjust(hspace = 0.05)
# ax[2].set_xlabel('rest-frame wavelength', fontsize = fsize)
# ax[1].set_ylabel('flux', fontsize = fsize)
# ax[0].tick_params(axis=u'both', direction='in', which='both')
# ax[1].tick_params(axis=u'both', direction='in', which='both')
# ax[2].tick_params(axis=u'both', direction='in', which='both')

# plt.savefig('plots/spectra.pdf', bbox_inches = 'tight')
#plt.close()








