#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 12 20:22:48 2021

@author: eilers
"""

import sys
sys.path.append('../RM_black_hole_masses')
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
from astropy.table import Column, Table, join, vstack, hstack
from astropy.convolution import convolve, Box1DKernel, Gaussian1DKernel
import pickle
import emcee
import corner
from astropy.io import fits
from scipy.optimize import minimize
import scipy.optimize as op
from astropy.stats import sigma_clip
from scipy.interpolate import UnivariateSpline
from _cont_src import fit_sdss_cont
from scipy.optimize import minimize

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

colors_cont = ["black", "grey", "light grey"] 
colors_cont = sns.xkcd_palette(colors_cont)

np.random.seed(40)


# -------------------------------------------------------------------------------
# To Do
# -------------------------------------------------------------------------------

'''
-- add VIS to X-Shooter data --> wavelength grid between 1240 to 2000 A in rest-frame
-- do we need complete wavelength grid or can we have np.nan for missing wavelengths?
-- missing values for Hbeta or MgII FWHM measurements?
-- check systematic uncertainty for BH measuremets derived from SE CIV or SE MgII lines
'''

# -------------------------------------------------------------------------------
# spectra class
# -------------------------------------------------------------------------------

class quasars:
       
    def __init__(self, name, z, MBH, MBH_err, M1450=None, M1450_err=None, Lbol=None, Lbol_err=None, L1350=None, L1350_err=None, SNR = None, survey = None, 
                 VP_Woo15 = None, VP_err_Woo15 = None, 
                 fwhm_MgII=None, fwhm_MgII_err=None, fwhm_CIV = None, fwhm_CIV_err = None, 
                 VIS_file = None, NIR_file = None, HST_file = None):
        
        self.name = name
        self.z = z
        self.M1450 = M1450
        self.M1450_err = M1450_err
        self.MBH = MBH
        self.MBH_err = MBH_err
        self.Lbol = Lbol # in 1e46 erg/s
        self.Lbol_err = Lbol_err
        self.L1350 = L1350
        self.L1350_err = L1350_err
        self.SNR = SNR
        self.fwhm_MgII = fwhm_MgII
        self.fwhm_MgII_err = fwhm_MgII_err
        self.fwhm_CIV = fwhm_CIV
        self.fwhm_CIV_err = fwhm_CIV_err
        self.VIS_file = VIS_file
        self.NIR_file = NIR_file
        self.HST_file = HST_file
        self.survey = survey
        self.VP_Woo15 = VP_Woo15
        self.VP_err_Woo15 = VP_err_Woo15
        
        if self.VIS_file: 
            xx = fits.getdata(self.VIS_file)
            try:
                self.wave_vis = xx['wave']
                self.flux_vis = xx['flux']      # 1e-17 erg/s/cm^2 ???
                self.ivar_vis = xx['ivar']  
            except:
                self.wave_vis = xx['OPT_WAVE']
                self.flux_vis = xx['OPT_FLAM']      # 1e-17 erg/s/cm^2 ???
                self.ivar_vis = xx['OPT_FLAM_IVAR']
        
        if self.NIR_file:
            xx = fits.getdata(self.NIR_file)
            self.wave_nir = xx['OPT_WAVE']
            self.flux_nir = xx['OPT_FLAM']      # 1e-17 erg/s/cm^2 ???
            self.ivar_nir = xx['OPT_FLAM_IVAR']
            
        if self.VIS_file and self.NIR_file:            
            ind_vis = self.wave_vis < 10100
            ind_nir = self.wave_nir >= 10100
            match_pix = 1000
            mean_flux_vis = np.median(convolve(self.flux_vis[ind_vis], Box1DKernel(500))[-match_pix:])
            mean_flux_nir = np.median(convolve(self.flux_nir[ind_nir], Box1DKernel(500))[:match_pix])
            correction = mean_flux_nir / mean_flux_vis
                       
            self.wave = np.hstack([self.wave_vis[ind_vis], self.wave_nir[ind_nir]])
            self.flux = np.hstack([self.flux_vis[ind_vis] * correction, self.flux_nir[ind_nir]])
            self.ivar = np.hstack([self.ivar_vis[ind_vis], self.ivar_nir[ind_nir] /correction**2])            
            self.wave_rest = self.wave / (1 + self.z)

        if self.z > 5: 
            atmos_mask = np.ones_like(self.wave_rest, dtype = bool)
            atmos_mask[(self.wave > 13450)*(self.wave < 14400)] = False
            atmos_mask[(self.wave > 18000)*(self.wave < 19400)] = False
            if self.name == 'PSOJ158-14':
                atmos_mask[(self.wave > 20600)] = False
            else:
                atmos_mask[(self.wave > 22500)] = False
            self.flux[~atmos_mask] = 0.0
            self.ivar[~atmos_mask] = 0.0
        
        # adapt to different geometry factor (Woo+15)
        # if self.z < 5:
        #     self.MBH = np.log10(10**self.MBH / 5.5 * 4.47)
        #     new_err = self.MBH_err**2 - 0.31**2 + 0.12**2 + 0.16**2
        #     self.MBH_err = np.sqrt(new_err)
            
            
        if self.HST_file:
            xx = np.loadtxt(self.HST_file)
            self.wave_rest = xx[:, 0]
            self.flux = xx[:, 1]           # 1e-14 erg/s/cm^2
            self.ivar = 1/xx[:, 2]**2
            if np.diff(self.wave_rest)[0] < 0:
                self.wave_rest = self.wave_rest[::-1]
                self.flux = self.flux[::-1]
                self.ivar = self.ivar[::-1]
            
        if self.L1350:
            # Richards+2006b (ApJS, 166, 470)
            self.log_Lbol = np.log10(3.81) + self.L1350
            self.log_Lbol_err = self.L1350_err
        
        # systematic uncertainty of 0.37 dex for SE derived BH masses -- Park+ 2017
        if self.z > 5: 
            self.MBH = np.log10(self.MBH * 1e9)
            self.MBH_err = self.MBH_err * 1e9 / (10**self.MBH) / np.log(10) 
            # add systematic uncertainty in quadrature? 
            self.MBH_err = np.sqrt(self.MBH_err**2 + 0.4**2) 
            
        if self.z > 5:
            self.log_Lbol = np.log10(self.Lbol * 1e46)
            self.log_Lbol_err = self.Lbol_err / self.Lbol / np.log(10)
        
        # add uncertainty of geometry factor (0.31 dex) for all quasars?
        # self.MBH_err = np.sqrt(self.MBH_err**2 + 0.31**2)
        
        if self.M1450:
            dL = planck.luminosity_distance(self.z).value * 1e6        # in pc
            m_AB = self.M1450 + 5 * (np.log10(dL) - 1) - 2.5 * np.log10(1+self.z)
            lam = 1450. * (1+self.z) * u.AA
            f_nu = 10**(m_AB / (-2.5)) * 3631 * u.Jy
            f_lam = f_nu*const.c/lam**2
            self.L1450 = np.log10((1450*u.AA*f_lam * 4*np.pi*(dL*u.pc)**2*(1+self.z)).to(u.erg/u.s).value)
        #     # NEEDS FIXING
        #     self.L1450_err = 0.005
        #     self.Lbol = 0.75 * (4.74 + 0.91 * self.L1450)
        #     self.Lbol_err = 0.75 * np.sqrt(1.**2 + (self.L1450 * 0.02)**2 + (0.91 * self.L1450_err)**2)
            

# -------------------------------------------------------------------------------
# low-z HST
# -------------------------------------------------------------------------------

qsos = {}

qsos['3C120'] = quasars(name = '3C120', z = 0.03301, MBH = 7.80, MBH_err = 0.04, L1350 = 44.399, L1350_err = 0.021, SNR = 12, survey = 'HST', 
                        VP_Woo15 = 12.2, VP_err_Woo15 = 0.9, fwhm_CIV= 3093, fwhm_CIV_err=291, 
                        HST_file = '/Users/eilers/Dropbox/projects/RM_black_hole_masses/specdata/Park_13data/HST_IUE_spectra_3C120.txt')

# qsos['3C390'] = quasars(name = '3C390', z = 0.05610, MBH = 8.43, MBH_err = 0.10, L1350=43.869, L1350_err=0.003, SNR = 18, survey = 'HST', # should be: 8.38
#                         VP_Woo15 = 278.1, VP_err_Woo15 = (24.4+31.6)/2, fwhm_CIV=5645, fwhm_CIV_err=202,
#                             HST_file = '/Users/eilers/Dropbox/projects/RM_black_hole_masses/specdata/Park_13data/HST_IUE_spectra_3C390.3.txt')

qsos['Ark120'] = quasars(name = 'Ark120', z = 0.03230, MBH = 8.14, MBH_err = 0.06, L1350=44.400, L1350_err=0.005, SNR = 17, survey = 'HST', 
                        VP_Woo15 = 23.7, VP_err_Woo15 = (3.0+4.2)/2., fwhm_CIV=3471, fwhm_CIV_err=108,
                            HST_file = '/Users/eilers/Dropbox/projects/RM_black_hole_masses/specdata/Park_13data/HST_IUE_spectra_Ark120.txt')

qsos['Fairall9'] = quasars(name = 'Fairall9', z = 0.04702, MBH = 8.38, MBH_err = 0.10, L1350=44.442, L1350_err=0.004, SNR = 24, survey = 'HST', #should be: 8.40
                        fwhm_CIV=2649, fwhm_CIV_err=20,
                            HST_file = '/Users/eilers/Dropbox/projects/RM_black_hole_masses/specdata/Park_13data/HST_IUE_spectra_Fairall9.txt')

qsos['Mrk279'] = quasars(name = 'Mrk279', z = 0.03045, MBH = 7.51, MBH_err = 0.12, L1350=43.082, L1350_err=0.004, SNR = 9, survey = 'HST', # should be: 7.51
                        VP_Woo15 = 7.2, VP_err_Woo15 = 0.8, fwhm_CIV=4093, fwhm_CIV_err=388,
                            HST_file = '/Users/eilers/Dropbox/projects/RM_black_hole_masses/specdata/Park_13data/HST_IUE_spectra_Mrk279.txt')

qsos['Mrk290'] = quasars(name = 'Mrk290', z = 0.02958, MBH = 7.36, MBH_err = 0.07, L1350=43.611, L1350_err=0.002, SNR = 24, survey = 'HST', 
                        fwhm_CIV=2052, fwhm_CIV_err=36,
                            HST_file = '/Users/eilers/Dropbox/projects/RM_black_hole_masses/specdata/Park_13data/HST_IUE_spectra_Mrk290.txt')

qsos['Mrk335'] = quasars(name = 'Mrk335', z = 0.02578, MBH = 7.37, MBH_err = 0.05, L1350=43.953, L1350_err=0.001, SNR = 29, survey = 'HST', 
                        fwhm_CIV=1772, fwhm_CIV_err=14,
                            HST_file = '/Users/eilers/Dropbox/projects/RM_black_hole_masses/specdata/Park_13data/HST_IUE_spectra_Mrk335.txt')

qsos['Mrk509'] = quasars(name = 'Mrk509', z = 0.03440, MBH = 8.12, MBH_err = 0.04, L1350=44.675, L1350_err=0.001, SNR = 107, survey = 'HST', 
                        VP_Woo15 = 22.2, VP_err_Woo15 = 0.7, fwhm_CIV=3872, fwhm_CIV_err=18, 
                            HST_file = '/Users/eilers/Dropbox/projects/RM_black_hole_masses/specdata/Park_13data/HST_IUE_spectra_Mrk509.txt')

qsos['Mrk590'] = quasars(name = 'Mrk590', z = 0.02638, MBH = 7.65, MBH_err = 0.07, L1350=44.094, L1350_err=0.007, SNR = 17, survey = 'HST', 
                        VP_Woo15 = 8.8, VP_err_Woo15 = (0.6+1.1)/2, fwhm_CIV=5362, fwhm_CIV_err=266,
                            HST_file = '/Users/eilers/Dropbox/projects/RM_black_hole_masses/specdata/Park_13data/HST_IUE_spectra_Mrk590.txt')

qsos['Mrk817'] = quasars(name = 'Mrk817', z = 0.03145, MBH = 7.66, MBH_err = 0.07, L1350=44.326, L1350_err=0.001, SNR = 38, survey = 'HST', 
                        VP_Woo15 = 12.6, VP_err_Woo15 = 1.2, fwhm_CIV=4580, fwhm_CIV_err=48,
                            HST_file = '/Users/eilers/Dropbox/projects/RM_black_hole_masses/specdata/Park_13data/HST_IUE_spectra_Mrk817.txt')

qsos['NGC3516'] = quasars(name = 'NGC3516', z = 0.00884, MBH = 7.47, MBH_err = 0.05, L1350=42.615, L1350_err=0.002, SNR = 20, survey = 'HST', 
                        VP_Woo15 = 7.2, VP_err_Woo15 = 0.6, fwhm_CIV=2658, fwhm_CIV_err=34,
                            HST_file = '/Users/eilers/Dropbox/projects/RM_black_hole_masses/specdata/Park_13data/HST_IUE_spectra_NGC3516.txt')

qsos['NGC3783'] = quasars(name = 'NGC3783', z = 0.00973, MBH = 7.44, MBH_err = 0.08, L1350=43.400, L1350_err=0.001, SNR = 29, survey = 'HST', 
                        VP_Woo15 = 4.4, VP_err_Woo15 = 0.6, fwhm_CIV=2656, fwhm_CIV_err=444,
                            HST_file = '/Users/eilers/Dropbox/projects/RM_black_hole_masses/specdata/Park_13data/HST_IUE_spectra_NGC3783.txt')

qsos['NGC4051'] = quasars(name = 'NGC4051', z = 0.00234, MBH = 6.20, MBH_err = 0.14, L1350=41.187, L1350_err=0.001, SNR = 23, survey = 'HST', 
                          VP_Woo15 = 0.4, VP_err_Woo15 = 0.04, fwhm_CIV = 1122, fwhm_CIV_err=309, 
                            HST_file = '/Users/eilers/Dropbox/projects/RM_black_hole_masses/specdata/Park_13data/HST_IUE_spectra_NGC4051.txt')

qsos['NGC4593'] = quasars(name = 'NGC4593', z = 0.00900, MBH = 6.96, MBH_err = 0.09, L1350=43.761, L1350_err=0.005, SNR = 10, survey = 'HST', 
                        VP_Woo15 = 2.1, VP_err_Woo15 = 0.3, fwhm_CIV=2952, fwhm_CIV_err=166,
                            HST_file = '/Users/eilers/Dropbox/projects/RM_black_hole_masses/specdata/Park_13data/HST_IUE_spectra_NGC4593.txt')

qsos['NGC5548'] = quasars(name = 'NGC5548', z = 0.01717, MBH = 7.80, MBH_err = 0.14, L1350=43.822, L1350_err=0.001, SNR = 36, survey = 'HST', 
                        VP_Woo15 = 16.3, VP_err_Woo15 = 2.5, fwhm_CIV=1785, fwhm_CIV_err=82,
                            HST_file = '/Users/eilers/Dropbox/projects/RM_black_hole_masses/specdata/Park_13data/HST_IUE_spectra_NGC5548.txt')

qsos['NGC7469'] = quasars(name = 'NGC7469', z = 0.01632, MBH = 7.05, MBH_err = 0.05, L1350=43.909, L1350_err=0.001, SNR = 32, survey = 'HST', 
                        VP_Woo15 = 4.8, VP_err_Woo15 = 1.0, fwhm_CIV=2725, fwhm_CIV_err=66,
                            HST_file = '/Users/eilers/Dropbox/projects/RM_black_hole_masses/specdata/Park_13data/HST_IUE_spectra_NGC7469.txt')

qsos['PG0026+129'] = quasars(name = 'PG0026+129', z = 0.14200, MBH = 8.56, MBH_err = 0.11, L1350=45.236, L1350_err=0.005, SNR = 25, survey = 'HST', 
                        fwhm_CIV=1604, fwhm_CIV_err=50,
                            HST_file = '/Users/eilers/Dropbox/projects/RM_black_hole_masses/specdata/Park_13data/HST_IUE_spectra_PG0026+129.txt')

qsos['PG0052+251'] = quasars(name = 'PG0052+251', z = 0.15500, MBH = 8.54, MBH_err = 0.09, L1350=45.292, L1350_err=0.004, SNR = 21, survey = 'HST', 
                        fwhm_CIV=5380, fwhm_CIV_err=87,
                            HST_file = '/Users/eilers/Dropbox/projects/RM_black_hole_masses/specdata/Park_13data/HST_IUE_spectra_PG0052+251.txt')

qsos['PG0804+761'] = quasars(name = 'PG0804+761', z = 0.10000, MBH = 8.81, MBH_err = 0.05, L1350=45.493, L1350_err=0.001, SNR = 34, survey = 'HST', 
                        fwhm_CIV=3429, fwhm_CIV_err=23,
                            HST_file = '/Users/eilers/Dropbox/projects/RM_black_hole_masses/specdata/Park_13data/HST_IUE_spectra_PG0804+761.txt')

qsos['PG0953+414'] = quasars(name = 'PG0953+414', z = 0.23410, MBH = 8.41, MBH_err = 0.09, L1350=45.629, L1350_err=0.005, SNR = 18, survey = 'HST', 
                        fwhm_CIV=3021, fwhm_CIV_err=74,
                            HST_file = '/Users/eilers/Dropbox/projects/RM_black_hole_masses/specdata/Park_13data/HST_IUE_spectra_PG0953+414.txt')

qsos['PG1226+023'] = quasars(name = 'PG1226+023', z = 0.15830, MBH = 8.92, MBH_err = 0.09, L1350=46.309, L1350_err=0.001, SNR = 93, survey = 'HST', 
                        fwhm_CIV=3609, fwhm_CIV_err=29,
                            HST_file = '/Users/eilers/Dropbox/projects/RM_black_hole_masses/specdata/Park_13data/HST_IUE_spectra_PG1226+023.txt')

qsos['PG1229+204'] = quasars(name = 'PG1229+204', z = 0.06301, MBH = 7.83, MBH_err = 0.23, L1350=44.609, L1350_err=0.009, SNR = 28, survey = 'HST', 
                        fwhm_CIV=4023, fwhm_CIV_err=163,
                            HST_file = '/Users/eilers/Dropbox/projects/RM_black_hole_masses/specdata/Park_13data/HST_IUE_spectra_PG1229+204.txt')

qsos['PG1307+085'] = quasars(name = 'PG1307+085', z = 0.15500, MBH = 8.61, MBH_err = 0.12, L1350=45.113, L1350_err=0.006, SNR = 14, survey = 'HST', 
                        fwhm_CIV=3604, fwhm_CIV_err=111,
                            HST_file = '/Users/eilers/Dropbox/projects/RM_black_hole_masses/specdata/Park_13data/HST_IUE_spectra_PG1307+085.txt')

qsos['PG1426+015'] = quasars(name = 'PG1426+015', z = 0.08647, MBH = 9.08, MBH_err = 0.13, L1350=45.263, L1350_err=0.004, SNR = 45, survey = 'HST', 
                        VP_Woo15 = 373.6, VP_err_Woo15 = (49.9+53.8)/2, fwhm_CIV=4220, fwhm_CIV_err=258,
                            HST_file = '/Users/eilers/Dropbox/projects/RM_black_hole_masses/specdata/Park_13data/HST_IUE_spectra_PG1426+015.txt')

qsos['PG1613+658'] = quasars(name = 'PG1613+658', z = 0.12900, MBH = 8.42, MBH_err = 0.22, L1350=45.488, L1350_err=0.001, SNR = 37, survey = 'HST', 
                        fwhm_CIV=6398, fwhm_CIV_err=51,
                            HST_file = '/Users/eilers/Dropbox/projects/RM_black_hole_masses/specdata/Park_13data/HST_IUE_spectra_PG1613+658.txt')

qsos['PG2130+099'] = quasars(name = 'PG2130+099', z = 0.06298, MBH = 7.63, MBH_err = 0.04, L1350=44.447, L1350_err=0.001, SNR = 22, survey = 'HST', 
                        VP_Woo15 = 20.1, VP_err_Woo15 = 2.8, fwhm_CIV=2147, fwhm_CIV_err=18,
                            HST_file = '/Users/eilers/Dropbox/projects/RM_black_hole_masses/specdata/Park_13data/HST_IUE_spectra_PG2130+099.txt')

qsos['Arp151'] = quasars(name = 'Arp151', z = 0.02109, MBH = 6.83, MBH_err = 0.08, L1350=41.791, L1350_err=0.017, SNR = 6, survey = 'HST', 
                        VP_Woo15 = 1.2, VP_err_Woo15 = 0.15, fwhm_CIV=1489, fwhm_CIV_err=26,
                            HST_file = '/Users/eilers/Dropbox/projects/RM_black_hole_masses/specdata/Park_17data/HST_STIS_spectra_arp151.txt')

qsos['Mrk1310'] = quasars(name = 'Mrk1310', z = 0.01956, MBH = 6.50, MBH_err = 0.14, L1350=41.715, L1350_err=0.025, SNR = 5, survey = 'HST', 
                        VP_Woo15 = 0.7, VP_err_Woo15 = 0.15, fwhm_CIV=1434, fwhm_CIV_err=78,
                            HST_file = '/Users/eilers/Dropbox/projects/RM_black_hole_masses/specdata/Park_17data/HST_STIS_spectra_mrk1310.txt')

qsos['Mrk50'] = quasars(name = 'Mrk50', z = 0.02343, MBH = 7.50, MBH_err = 0.08, L1350=43.213, L1350_err=0.003, SNR = 19, survey = 'HST', 
                        VP_Woo15 = 6.3, VP_err_Woo15 = 0.7, fwhm_CIV=2807, fwhm_CIV_err=63,
                            HST_file = '/Users/eilers/Dropbox/projects/RM_black_hole_masses/specdata/Park_17data/HST_STIS_spectra_mrk50.txt')

qsos['NGC6814'] = quasars(name = 'NGC6814', z = 0.00521, MBH = 7.28, MBH_err = 0.14, L1350=41.105, L1350_err=0.021, SNR = 6, survey = 'HST', 
                        VP_Woo15 = 4.2, VP_err_Woo15 = 0.8, fwhm_CIV=2651, fwhm_CIV_err=264,
                            HST_file = '/Users/eilers/Dropbox/projects/RM_black_hole_masses/specdata/Park_17data/HST_STIS_spectra_ngc6814.txt')

qsos['SBS1116+583A'] = quasars(name = 'SBS1116+583A', z = 0.02787, MBH = 7.74, MBH_err = 0.22, L1350=42.867, L1350_err=0.005, SNR = 13, survey = 'HST', 
                        VP_Woo15 = 1.1, VP_err_Woo15 = 0.5, fwhm_CIV=3253, fwhm_CIV_err=302,
                            HST_file = '/Users/eilers/Dropbox/projects/RM_black_hole_masses/specdata/Park_17data/HST_STIS_spectra_sbs1116.txt')

qsos['Zw229-015'] = quasars(name = 'Zw229-015', z = 0.02788, MBH = 6.99, MBH_err = 0.08, L1350=43.129, L1350_err=0.007, SNR = 17, survey = 'HST', 
                        fwhm_CIV=2573, fwhm_CIV_err=71,
                            HST_file = '/Users/eilers/Dropbox/projects/RM_black_hole_masses/specdata/Park_17data/HST_STIS_spectra_zw229.txt')



# -------------------------------------------------------------------------------
# virial factor
# -------------------------------------------------------------------------------

# Woo et al. 2010
log_f_applied  = 0.71
#log_f_err_applied = 0.31 # already subtracted from Park+17 measurements (and not included in Park+13 measurements)

# Woo et al. 2015
log_f_new = 0.65
log_f_err = 0.12


for i, q in enumerate(qsos.keys()):
    #qsos[q].logVP = qsos[q].MBH - log_f_applied
    #qsos[q].logVP_err = qsos[q].MBH_err
    qsos[q].MBH = qsos[q].MBH + log_f_new - log_f_applied
    qsos[q].MBH_err = np.sqrt(qsos[q].MBH_err**2 + log_f_err**2)
    # try:
    #     qsos[q].logVP_Woo15 = np.log10(qsos[q].VP_Woo15*1e6)
    #     qsos[q].logVP_err_Woo15 = qsos[q].VP_err_Woo15 / qsos[q].VP_Woo15 / np.log(10)
    # except:
    #     qsos[q].logVP_Woo15 = None
    #     qsos[q].logVP_err_Woo15 = None
        
# -------------------------------------------------------------------------------
# xxx
# -------------------------------------------------------------------------------

# for i, q in enumerate(qsos.keys()):
#     print(qsos[q].name, qsos[q].logVP_Woo15, qsos[q].logVP, qsos[q].MBH)

    
# -------------------------------------------------------------------------------
# wavelength grid
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

# dline = 60.
# dlam_noline = 20.
# dlam_line = 2.

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

# wave_grid = np.array(wave_grid)
# print('number of pixels: {}'.format(len(wave_grid)))

wave_grid = np.arange(min_waves, max_waves, 2)
data = np.zeros((len(qsos), len(wave_grid)+5)) 
data_ivar = np.zeros((len(qsos), len(wave_grid)+5)) 

# -------------------------------------------------------------------------------
# normalize to around unity 
# -------------------------------------------------------------------------------

def PowerLaw(F0, alpha, wl):    
    return F0 * (wl/2500.)**alpha

def lnlike(pars, flux, ivar, wl):
    F0, alpha = pars
    return np.sum((flux - PowerLaw(F0, alpha, wl))**2 * ivar)

# -------------------------------------------------------------------------------
# mask absorption lines
# -------------------------------------------------------------------------------

# mask infinite values
for i, q in enumerate(qsos.keys()):
    
    masks = np.zeros_like(qsos[q].flux, dtype = bool)
    masks[np.isfinite(qsos[q].wave_rest) * np.isfinite(qsos[q].flux)] = True
    qsos[q].flux = qsos[q].flux[masks]
    qsos[q].wave_rest = qsos[q].wave_rest[masks]
    qsos[q].ivar = qsos[q].ivar[masks]

# mask absoprtion lines
for i, q in enumerate(qsos.keys()):
    qsos[q].cont = np.zeros_like(qsos[q].flux)
    nfit = len(qsos[q].flux)
    if qsos[q].name == '3C390' or qsos[q].name == 'Ark120' or qsos[q].name == 'Fairall9' or qsos[q].name == 'Mrk279' or qsos[q].name == 'Mrk290' or qsos[q].name == 'Mrk335' or qsos[q].name == 'Mrk817' or qsos[q].name == 'NGC4593' or qsos[q].name == 'NGC7469' or qsos[q].name == 'PG0804+761' or qsos[q].name == 'PG1226+023' or qsos[q].name == 'PG1307+085' or qsos[q].name == 'PG1613+658' or qsos[q].name == 'PG2130+099':
        deltapix2 = 17
    elif qsos[q].name == 'Mrk509' or qsos[q].name == 'NGC3516' or qsos[q].name == 'NGC3783' or qsos[q].name == 'NGC5548' or qsos[q].name == 'PG0052+251' or qsos[q].name == 'PG0953+414' or qsos[q].name == 'NGC4051':
        deltapix2 = 40
    elif qsos[q].name == 'PG0026+129' or qsos[q].name == 'PG1426+015' or qsos[q].name == 'Mrk50' or qsos[q].name == 'NGC6814' or qsos[q].name == 'SBS1116+583A': # or qsos[q].name == 'Zw229-015':
        deltapix2 = 8
    else:
        deltapix2 = 5
    fit_sdss_cont(qsos[q].wave_rest, qsos[q].flux, 1./np.sqrt(qsos[q].ivar), nfit, 0, 20, deltapix2, 4, 0.033, 0.99, qsos[q].cont, 1.0)
    xx = sigma_clip((qsos[q].flux-qsos[q].cont), sigma_lower = 3, sigma_upper = 4) 
    # plt.plot(qsos[q].wave_rest, qsos[q].flux, color = '0.8')
    # plt.plot(qsos[q].wave_rest[~xx.mask], qsos[q].flux[~xx.mask], color = 'k', label = '{}, dpix={}'.format(qsos[q].name, deltapix2))
    # plt.plot(qsos[q].wave_rest, qsos[q].cont, color = 'r', zorder = 10)
    # plt.xlim(1220, 5000)
    # plt.ylim(0, 30)
    # plt.legend()
    qsos[q].flux_orig = qsos[q].flux.copy()
    qsos[q].wave_orig = qsos[q].wave_rest.copy()
    qsos[q].flux = qsos[q].flux[~xx.mask]
    qsos[q].ivar = qsos[q].ivar[~xx.mask]
    qsos[q].wave_rest = qsos[q].wave_rest[~xx.mask]    

# normalize to "something like the mean continuum"
for i, q in enumerate(qsos.keys()):

    # # mask emission lines
    # inds = (qsos[q].wave_rest > min_waves) * (qsos[q].wave_rest < max_waves)
    # yy = convolve(qsos[q].flux[inds], Box1DKernel(500))
    # xx = sigma_clip((qsos[q].flux[inds]-yy)*np.sqrt(qsos[q].ivar[inds]), sigma_lower = 1, sigma_upper = 1)
    # fig = plt.subplots(1, 1, figsize = (12, 6))
    # plt.plot(qsos[q].wave_rest[inds], qsos[q].flux[inds])
    # #plt.plot(qsos[q].wave_rest, yy)
    # plt.plot(qsos[q].wave_rest[inds][~xx.mask], qsos[q].flux[inds][~xx.mask])
    # mean = np.nanmean(qsos[q].flux[inds][~xx.mask])
    # print(i, mean)
    # qsos[q].flux_norm = qsos[q].flux / mean
    # qsos[q].ivar_norm = qsos[q].ivar * mean**2
    # plt.axhline(mean)
    # plt.close()
       
    # -------------------------------------------------------------------------------
    # fit power-law
    # -------------------------------------------------------------------------------

    # areas free of emission lines
    good_indices = np.zeros_like(qsos[q].wave_rest, dtype = bool)
    good_indices[(qsos[q].wave_rest > 1270) * (qsos[q].wave_rest < 1280)] = True
    good_indices[(qsos[q].wave_rest > 1350) * (qsos[q].wave_rest < 1370)] = True
    good_indices[(qsos[q].wave_rest > 1430) * (qsos[q].wave_rest < 1470)] = True
    good_indices[(qsos[q].wave_rest > 1600) * (qsos[q].wave_rest < 1620)] = True
    good_indices[(qsos[q].wave_rest > 1690) * (qsos[q].wave_rest < 1730)] = True
    good_indices[(qsos[q].wave_rest > 1770) * (qsos[q].wave_rest < 1850)] = True
    good_indices[(qsos[q].wave_rest > 1950) * (qsos[q].wave_rest < 2700)] = True
    good_indices[(qsos[q].wave_rest > 2900) * (qsos[q].wave_rest < 3400)] = True
    good_indices[(qsos[q].wave_rest > 3450) * (qsos[q].wave_rest < 3700)] = True
    good_indices[(qsos[q].wave_rest > 3750) * (qsos[q].wave_rest < 3850)] = True
    good_indices[(qsos[q].wave_rest > 3900) * (qsos[q].wave_rest < 4300)] = True
    good_indices[(qsos[q].wave_rest > 4400) * (qsos[q].wave_rest < 4800)] = True
    
    x0 = np.array([1, -1])
    bnds = [(0, None), (-4, 0)]
    res = minimize(lnlike, x0 = x0, args = (qsos[q].flux[good_indices], qsos[q].ivar[good_indices], qsos[q].wave_rest[good_indices]), bounds = bnds)
    F0, qsos[q].alpha = res.x
    # normalization 
    mean = PowerLaw(F0, qsos[q].alpha, 2500)
    print(i, mean)
    qsos[q].flux_norm = qsos[q].flux / mean
    qsos[q].ivar_norm = qsos[q].ivar * mean**2
    
    # -------------------------------------------------------------------------------
    # normalize to unity at ~1280 A
    # -------------------------------------------------------------------------------
    #dl = 5
    # wave_norm = 1500 
    # inds = (qsos[q].wave_rest > (wave_norm - dl)) * (qsos[q].wave_rest < (wave_norm + dl))
    # mean = np.nanmean(qsos[q].flux[inds])
    # # if wave_norm < qsos[q].wave_rest[0] or np.isnan(mean) or mean == 0:
    # #     wave_norm = 2500
    # #     inds = (qsos[q].wave_rest > (wave_norm - dl)) * (qsos[q].wave_rest < (wave_norm + dl))
    # #     mean = np.nanmean(qsos[q].flux[inds])
    # # if wave_norm < qsos[q].wave_rest[0] or np.isnan(mean) or mean == 0:
    # #     wave_norm = 3500
    # #     inds = (qsos[q].wave_rest > (wave_norm - dl)) * (qsos[q].wave_rest < (wave_norm + dl))
    # #     mean = np.nanmean(qsos[q].flux[inds])
    # print(i, mean)
    # qsos[q].flux_norm = qsos[q].flux / mean
    # qsos[q].ivar_norm = qsos[q].ivar * mean**2
    
    # interpolate to new wavelengths
    qsos[q].wave_stack, qsos[q].flux_stack, qsos[q].ivar_stack, mask_stack, nused = compute_stack(wave_grid, qsos[q].wave_rest, qsos[q].flux_norm, qsos[q].ivar_norm, 
                                                                                                  np.ones_like(qsos[q].wave_rest, dtype=bool), np.ones_like(qsos[q].wave_rest))    

    data[i, 6:] = qsos[q].flux_stack
    data_ivar[i, 6:] = qsos[q].ivar_stack
    
    # missing data   
    data[i, 6:][qsos[q].flux_stack == 0] = np.nan
    data_ivar[i, 6:][np.isinf(qsos[q].ivar_stack)] = np.nan
    
    data[i, 0] = qsos[q].MBH 
    data_ivar[i, 0] = 1 / (qsos[q].MBH_err**2 + log_f_err**2)
    data[i, 1] = qsos[q].z
    data_ivar[i, 1] = 0.0001**(-2.)
    data[i, 2] = qsos[q].SNR
    data_ivar[i, 2] = 0
    data[i, 3] = qsos[q].alpha
    data_ivar[i, 3] = np.nan
    data[i, 4] = 5
    data_ivar[i, 4] = np.nan
    data[i, 5] = mean
    data_ivar[i, 5] = np.nan
    #data[i, 5] = qsos[q].fwhm_CIV
    #data_ivar[i, 5] = 1 / (qsos[q].fwhm_CIV_err**2)
    data[i, 6] = qsos[q].log_Lbol
    data_ivar[i, 6] = 1 / (qsos[q].log_Lbol_err**2)
    # try:
    #     data[i, 6] = qsos[q].L1350
    #     data_ivar[i, 6] = 1 / (qsos[q].L1350_err**2)
    # except:
    #     data[i, 6] = qsos[q].log_L1450
    #     data_ivar[i, 6] = 1 / (qsos[q].log_L1450_err**2 )       

    fig = plt.figure(figsize = (12, 6))
    plt.plot(qsos[q].wave_orig, qsos[q].flux_orig / mean, color = '0.9', label = r'original data: {}, {}, $z={}$, SNR=${}$, $\log M_\bullet = {}$'.format(i, qsos[q].name, qsos[q].z, qsos[q].SNR, np.round(qsos[q].MBH, 2)))
    plt.plot(qsos[q].wave_orig, qsos[q].cont / mean, color = 'r', zorder = 10, label = 'spline fit', lw = 2)
    plt.plot(qsos[q].wave_stack, data[i, 6:], color = 'k', label = 'input data', zorder  = 12)
    plt.plot(qsos[q].wave_stack, 1/np.sqrt(data_ivar[i, 6:]), color = '0.7', label = 'input noise')
    plt.plot(qsos[q].wave_stack, PowerLaw(F0, qsos[q].alpha, qsos[q].wave_stack) / mean, label = r'power law, $\alpha={}$'.format(round(qsos[q].alpha, 2)), lw = 1, linestyle = '--', color = 'blue')
    plt.ylim(-1, 12)
    plt.xlim(1190, 5000.1)
    for l in list(lines):
        plt.axvline(l)
    #plt.axhline(mean, color = colors[5], lw = 2)
    plt.legend(fontsize = 16)
    plt.savefig('../RM_black_hole_masses/BOSS/plots/spectrum_HST_{}.pdf'.format(i))
    plt.show()
    
# -------------------------------------------------------------------------------
# add missing data by setting missing values to composite spectrum?
# -------------------------------------------------------------------------------


# -------------------------------------------------------------------------------
# save files
# -------------------------------------------------------------------------------

f = open('../RM_black_hole_masses/data_HST_1220_5000_2A.pickle', 'wb')
pickle.dump([data, data_ivar], f)
f.close()

# -------------------------------------------------------------------------------'''




# -------------------------------------------------------------------------------'''
# X-Shooter surge -- Schindler+2020
# -------------------------------------------------------------------------------

'''xx = Table.read('Schindler20.txt', format = 'ascii.cds') # all different BH masses from CIV and MgII etc. 

cuts = np.log10(xx['Lbol']*1e46) < 47
xx[cuts]['Name', 'zsys', 'SNR-J', 'Lbol', 'E_Lbol', 'e_Lbol', 'CIV-BHM-Co17','e_CIV-BHM-Co17','E_CIV-BHM-Co17', 'VW01-MgII-BHM-S11', 'e_VW01-MgII-BHM-S11', 'E_VW01-MgII-BHM-S11'].pprint(max_width = -1)


# qsos['J0046-2837'] = quasars(name = 'J0046-2837', z = 5.9926, M1450 = -25.09, M1450_err = (0.19+0.24)/2, MBH = 1e9, MBH_err = 1e8, Lbol = 8.39, Lbol_err = 0.44,
#                              fwhm_MgII = 1737, fwhm_MgII_err = (88+79)/2,  
#                             NIR_file = '/Users/eilers/Dropbox/XShooter/surge/J0046-2837/NIR/spec1d_coadd_J0046-2837_tellcorr_nir_01.fits')

qsos['J0842+1218'] = quasars(name = 'J0842+1218', z = 6.0754, M1450 = -26.69, M1450_err = 0.01, MBH = 1.80, MBH_err = 0.075, Lbol = 19.17, Lbol_err = 0.215, SNR = 10.7, survey = 'X-Shooter', 
                            fwhm_MgII = 2935, fwhm_MgII_err = (131+123)/2,
                            fwhm_CIV=6027, fwhm_CIV_err=(135+137)/2, 
                            VIS_file = '/Users/eilers/Dropbox/XShooter/surge/J0842+1218/VIS/J0842_coadd.fits', 
                            NIR_file = '/Users/eilers/Dropbox/XShooter/surge/J0842+1218/NIR/spec1d_coadd_J0842+1218_tellcorr_nir_final_02.fits')

#qsos['J1207+0630'] = quasars(name = 'J1207+0630', z = 6, MBH = 1e9, MBH_err = 1e8, L = 1e46, fwhm_MgII = 400, 
#                            NIR_file = '/Users/eilers/Dropbox/XShooter/surge/J1207+0630/NIR/spec1d_coadd_J1207+0630_tellcorr_nir_final_02.fits')

qsos['J1306+0356'] = quasars(name = 'J1306+0356', z = 6.0330, M1450 = -26.70, M1450_err = 0.01, MBH = 2.33, MBH_err = 0.06, Lbol = 18.34, Lbol_err = 0.13, SNR = 18.6, survey = 'X-Shooter', 
                              fwhm_MgII = 3107, fwhm_MgII_err = (73+74)/2,
                              fwhm_CIV=5236, fwhm_CIV_err=(83+99)/2, 
                              VIS_file = '/Users/eilers/Dropbox/XShooter/surge/J1306+0356/VIS/J1306_coadd.fits', 
                              NIR_file = '/Users/eilers/Dropbox/XShooter/surge/J1306+0356/NIR/spec1d_coadd_J1306+0356_tellcorr_nir_final_02.fits')

qsos['J1319+0950'] = quasars(name = 'J1319+0950', z = 6.1347, M1450 = -26.80, M1450_err = 0.01, MBH = 2.86, MBH_err = 0.105, Lbol = 17.75, Lbol_err = 0.065, SNR = 22.8, survey = 'X-Shooter',
                              fwhm_MgII = 3155, fwhm_MgII_err=(138+131)/2, 
                              fwhm_CIV=8933, fwhm_CIV_err=(118+110)/2, 
                              VIS_file = '/Users/eilers/Dropbox/XShooter/surge/J1319+0950/VIS/J1319_coadd.fits', 
                              NIR_file = '/Users/eilers/Dropbox/XShooter/surge/J1319+0950/NIR/spec1d_coadd_J1319+0950_tellcorr_nir_05.fits')

qsos['J1509-1749'] = quasars(name = 'J1509-1749', z = 6.1225, M1450 = -26.56, M1450_err = 0.01, MBH = 2.49, MBH_err = 0.17, Lbol = 24.95, Lbol_err = 0.25, SNR = 11, survey = 'X-Shooter',
                              fwhm_MgII = 3491, fwhm_MgII_err=(191+171)/2, 
                              fwhm_CIV=5537, fwhm_CIV_err=(183+175)/2,
                              VIS_file = '/Users/eilers/Dropbox/XShooter/surge/J1509-1749/VIS/CFHQSJ1509_coadd.fits', 
                              NIR_file = '/Users/eilers/Dropbox/XShooter/surge/J1509-1749/NIR/spec1d_coadd_J1509-1749_tellcorr_nir_01.fits')

qsos['J2211-3206'] = quasars(name = 'J2211-3206', z = 6.3394, M1450 = -27.09, M1450_err = 0.03, MBH = 1.22, MBH_err = 0.10, Lbol = 29.45, Lbol_err = 0.59, SNR = 5.2, survey = 'X-Shooter',
                              fwhm_MgII = 3890, fwhm_MgII_err=(191+166)/2, 
                              fwhm_CIV=3996, fwhm_CIV_err=(250+246)/2,
                              VIS_file = '/Users/eilers/Dropbox/XShooter/surge/Feige_VIS/J2211-3206_XSHOOTER_VIS.fits', 
                              NIR_file = '/Users/eilers/Dropbox/XShooter/surge/J2211-3206/NIR/spec1d_coadd_J2211-3206_tellcorr_nir_final_02.fits')

# qsos['PSOJ007+04'] = quasars(name = 'PSOJ007+04', z = 6.0015, M1450 = -26.51, M1450_err = 0.055, MBH = 2.34, MBH_err = (0.63+0.9)/2, Lbol = 20.05, Lbol_err = 0.79,
#                             fwhm_MgII = 2781, fwhm_MgII_err = (1579+394)/2, 
#                             fwhm_CIV = 7278, fwhm_CIV_err = (1332+1090)/2, 
#                             VIS_file = '/Users/eilers/Dropbox/XShooter/surge/Feige_VIS/', 
#                             NIR_file = '/Users/eilers/Dropbox/XShooter/surge/PSOJ007+04/NIR/spec1d_coadd_J0028+0457_tellcorr_nir_01.fits')

#qsos['PSOJ009-10'] = quasars(name = 'PSOJ009-10', z = 6.0040, M1450 = -26.03, M1450_err = 0.04, MBH = 1e9, MBH_err = 1e8, L = 1e46, fwhm_MgII = ?, 
#                             fwhm_CIV = 15746, fwhm_CIV_err = (2315+2274)/2, 
#                            NIR_file = '/Users/eilers/Dropbox/XShooter/surge/PSOJ009-10/NIR/spec1d_coadd_J0038-1025_tellcorr_nir_final_02.fits')

qsos['PSOJ036+03'] = quasars(name = 'PSOJ036+03', z = 6.5405, M1450 = -27.15, M1450_err = 0.01, MBH = 2.77, MBH_err = (0.24+0.30)/2, Lbol = 24.89, Lbol_err = 0.145, SNR = 11.6, survey = 'X-Shooter',
                              fwhm_MgII = 3542, fwhm_MgII_err=(288+279)/2,  
                              fwhm_CIV=11640, fwhm_CIV_err=(557+496)/2, 
                              VIS_file = '/Users/eilers/Dropbox/XShooter/surge/Feige_VIS/J0226+0302_XSHOOTER_VIS.fits', 
                              NIR_file = '/Users/eilers/Dropbox/XShooter/surge/PSOJ036+03/NIR/spec1d_coadd_J0226+0302_tellcorr_nir_01.fits')

# qsos['PSOJ065-19'] = quasars(name = 'PSOJ065-19', z = 6.1247, M1450 = -26.11, M1450_err = 0.03, MBH = 1.14, MBH_err = 0.05, Lbol = 17.67, Lbol_err = 0.41,
#                               fwhm_MgII = 2830, fwhm_MgII_err=(87+80)/2, 
#                               fwhm_CIV=5638, fwhm_CIV_err=(245+215)/2,
#                               VIS_file = '/Users/eilers/Dropbox/XShooter/surge/PSOJ065-19/VIS/', 
#                               NIR_file = '/Users/eilers/Dropbox/XShooter/surge/PSOJ065-19/NIR/spec1d_coadd_J0422-1927_tellcorr_nir_final_02.fits')

qsos['PSOJ065-26'] = quasars(name = 'PSOJ065-26', z = 6.1871, M1450 = 26.94, M1450_err = 0.01, MBH = 0.90, MBH_err = 0.055, Lbol = 19.10, Lbol_err = 0.53, SNR = 8.7, survey = 'X-Shooter',
                              fwhm_MgII = 4032, fwhm_MgII_err=(216+192)/2, 
                              fwhm_CIV=7766, fwhm_CIV_err=(268+283)/2,
                              VIS_file = '/Users/eilers/Dropbox/XShooter/surge/PSOJ065-26/VIS/J065_coadd.fits', 
                              NIR_file = '/Users/eilers/Dropbox/XShooter/surge/PSOJ065-26/NIR/spec1d_coadd_J0421-2657_tellcorr_nir_02.fits')

qsos['PSOJ159-02'] = quasars(name = 'PSOJ159-02', z = 6.3809, M1450 = -26.47, M1450_err = 0.02, MBH = 1.38, MBH_err = 0.105, Lbol = 17.78, Lbol_err = 0.305, SNR = 4.9, survey = 'X-Shooter',
                              fwhm_MgII = 3297, fwhm_MgII_err=(235+208)/2,  
                              fwhm_CIV=4921, fwhm_CIV_err=(210+183)/2,
                              VIS_file = '/Users/eilers/Dropbox/XShooter/surge/Feige_VIS/J1036-0232_XSHOOTER_VIS.fits', 
                              NIR_file = '/Users/eilers/Dropbox/XShooter/surge/PSOJ159-02/NIR/spec1d_coadd_J1036-0232_tellcorr_nir_final_02.fits')

qsos['PSOJ183+05'] = quasars(name = 'PSOJ183+05', z = 6.4386, M1450 = -26.87, M1450_err = 0.015, MBH = 1.51, MBH_err = (0.25+0.37)/2, Lbol = 22.30, Lbol_err = 0.29, SNR = 6.1, survey = 'X-Shooter',
                              fwhm_MgII = 3132, fwhm_MgII_err=(259+263)/2, 
                              fwhm_CIV=8927, fwhm_CIV_err=(768+649)/2,
                              VIS_file = '/Users/eilers/Dropbox/XShooter/surge/Feige_VIS/J1212+0505_XSHOOTER_VIS.fits', 
                              NIR_file = '/Users/eilers/Dropbox/XShooter/surge/PSOJ183+05/NIR/spec1d_coadd_J1212+0505_tellcorr_nir_final_02.fits')

#qsos['PSOJ231-20'] = quasars(name = 'PSOJ231-20', z = 6.5869, M1450 = -27.07, M1450_err = 0.03, MBH = 1e9, MBH_err = 1e8, Lbol = 22.21, Lbol_err = 0.53,
#                             fwhm_MgII = 3894, fwhm_MgII_err=(569+585)/2, 
#                            NIR_file = '/Users/eilers/Dropbox/XShooter/surge/PSOJ231-20/NIR/spec1d_coadd_J1526-2050_tellcorr_nir_01.fits')

qsos['PSOJ323+12'] = quasars(name = 'PSOJ323+12', z = 6.5872, M1450 = -26.89, M1450_err = 0.01, MBH = 2.29, MBH_err = 0.10, Lbol = 19.81, Lbol_err = 0.21, SNR = 6.5, survey = 'X-Shooter',
                              fwhm_MgII = 2291, fwhm_MgII_err=(122+142)/2,
                              fwhm_CIV=3286, fwhm_CIV_err=(93+83)/2,
                              VIS_file = '/Users/eilers/Dropbox/XShooter/surge/Feige_VIS/J2132+1217_XShooter_VIS_tellcorr.fits', 
                              NIR_file = '/Users/eilers/Dropbox/XShooter/surge/PSOJ323+12/NIR/spec1d_coadd_J2132+1217_tellcorr_nir_02.fits')

qsos['VIKJ0109-3047'] = quasars(name = 'VIKJ0109-3047', z = 6.7904, M1450 = -26.89, M1450_err = 0.01, MBH = 1.53, MBH_err = 0.53, Lbol = 7.66, Lbol_err = 0.204, SNR = 2.6, survey = 'X-Shooter',
                              fwhm_MgII = 2291, fwhm_MgII_err=(122+142)/2,
                              fwhm_CIV=3286, fwhm_CIV_err=(93+83)/2,
                              VIS_file = '/Users/eilers/Dropbox/XShooter/surge/Feige_VIS/J2132+1217_XShooter_VIS_tellcorr.fits', 
                              NIR_file = '/Users/eilers/Dropbox/XShooter/surge/PSOJ323+12/NIR/spec1d_coadd_J2132+1217_tellcorr_nir_02.fits')

qsos['VIKJ2348-3054'] = quasars(name = 'VIKJ2348-3054', z = 6.9007, M1450 = -26.89, M1450_err = 0.01, MBH = 3.25, MBH_err = 1.05, Lbol = 7.46, Lbol_err = 0.209, SNR = 3.5, survey = 'X-Shooter',
                              fwhm_MgII = 2291, fwhm_MgII_err=(122+142)/2,
                              fwhm_CIV=3286, fwhm_CIV_err=(93+83)/2,
                              VIS_file = '/Users/eilers/Dropbox/XShooter/surge/Feige_VIS/J2132+1217_XShooter_VIS_tellcorr.fits', 
                              NIR_file = '/Users/eilers/Dropbox/XShooter/surge/PSOJ323+12/NIR/spec1d_coadd_J2132+1217_tellcorr_nir_02.fits')

# -------------------------------------------------------------------------------
# Eilers+2020
# -------------------------------------------------------------------------------

qsos['PSOJ004+17'] = quasars(name = 'PSOJ004+17', z = 5.8166, M1450 = -25.95, M1450_err = 0.045, MBH = 0.59, MBH_err = 0.125, Lbol = 6.21, Lbol_err = 0.205, SNR = 3.6, survey = 'X-Shooter',
                            fwhm_CIV = 4071, fwhm_CIV_err=(451+462)/2,  
                            VIS_file = '/Users/eilers/Dropbox/XShooter/Aug2018/PSOJ004+17/VIS/stack/spec1d_stack_PSOJ004+17_tellcorr.fits', 
                            NIR_file = '/Users/eilers/Dropbox/XShooter/Aug2018/PSOJ004+17/NIR/Science/spec1d_coadd_PSOJ004+17_tellcorr_nir.fits')


qsos['PSOJ056-16'] = quasars(name = 'PSOJ056-16', z = 5.9676, M1450 = -26.26, M1450_err = 0.02, MBH = 0.71, MBH_err = 0.04, Lbol = 10.52, Lbol_err = 0.12, SNR = 8.1, survey = 'X-Shooter',
                            fwhm_MgII = 2323, fwhm_MgII_err = (89+85)/2, 
                            fwhm_CIV = 2642, fwhm_CIV_err=(57+50)/2,  
                            VIS_file = '/Users/eilers/Dropbox/XShooter/PSOJ056-16/VIS/stack/spec1d_stack_PSOJ056-16_tellcorr.fits', 
                            NIR_file = '/Users/eilers/Dropbox/XShooter/PSOJ056-16/NIR/Science/spec1d_coadd_PSOJ056-16_tellcorr_nir.fits')


qsos['PSOJ158-14'] = quasars(name = 'PSOJ158-14', z = 6.0685, M1450 = -27.07, M1450_err = 0.03, MBH = 4.96, MBH_err = 0.51, Lbol = 46.61, Lbol_err = 1.105, SNR = 6.8, survey = 'X-Shooter',
                            fwhm_MgII = 2661, fwhm_MgII_err = (182+172)/2, 
                            fwhm_CIV = 7703, fwhm_CIV_err=(369+339)/2, 
                            VIS_file = '/Users/eilers/Dropbox/XShooter/PSOJ158-14/VIS/stack/spec1d_stack_PSOJ158-14_tellcorr.fits', 
                            NIR_file = '/Users/eilers/Dropbox/XShooter/PSOJ158-14/NIR2/Science/spec1d_coadd_J158-14_tellcorr_nir.fits')

qsos['J2100-1715'] = quasars(name = 'J2100-1715', z = 6.0806, M1450 = -24.63, M1450_err = 0.05, MBH = 2.18, MBH_err = 0.21, Lbol = 4.77, Lbol_err = 0.109, SNR = 4, survey = 'X-Shooter',
                            fwhm_MgII = 7726, fwhm_MgII_err = (1007+2572)/2, 
                            fwhm_CIV = 7433, fwhm_CIV_err=(2324+999)/2,  
                            VIS_file = '/Users/eilers/Dropbox/XShooter/J2100-1715/VIS/stack/spec1d_stack_J2100-1715_tellcorr.fits', 
                            NIR_file = '/Users/eilers/Dropbox/XShooter/J2100-1715/NIR/Science/spec1d_coadd_J2100-1715_tellcorr_nir.fits')

qsos['J2229+1457'] = quasars(name = 'J2229+1457', z = 6.1517, M1450 = -24.43, M1450_err = 0.075, MBH = 1.44, MBH_err = 0.25, Lbol = 2.86, Lbol_err = 0.16, SNR = 1.1, survey = 'X-Shooter',
                            fwhm_MgII = 5469, fwhm_MgII_err = 439, 
                            fwhm_CIV = 886, fwhm_CIV_err=(51+49)/2,  
                            VIS_file = '/Users/eilers/Dropbox/XShooter/Aug2018/J2229+1457/VIS/stack/spec1d_stack_J2229+1457_tellcorr.fits',
                            NIR_file = '/Users/eilers/Dropbox/XShooter/Aug2018/J2229+1457/NIR/Science/spec1d_coadd_J2229+1457_tellcorr_nir.fits')

qsos['PSOJ359-06'] = quasars(name = 'PSOJ359-06', z = 6.1722, M1450 = -26.62, M1450_err = 0.02, MBH = 1.05, MBH_err = 0.06, Lbol = 25.35, Lbol_err = 0.295, SNR = 5.3, survey = 'X-Shooter',
                            fwhm_MgII = 2505, fwhm_MgII_err = (240+171)/2, 
                            fwhm_CIV = 3520, fwhm_CIV_err=(123+117)/2,  
                            VIS_file = '/Users/eilers/Dropbox/XShooter/PSOJ359-06/VIS/stack/spec1d_stack_PSOJ359-06_tellcorr.fits', 
                            NIR_file = '/Users/eilers/Dropbox/XShooter/PSOJ359-06/NIR/Science/spec1d_coadd_PSOJ359-06_tellcorr_nir.fits')

# -------------------------------------------------------------------------------'''
# XQR-30
# -------------------------------------------------------------------------------


