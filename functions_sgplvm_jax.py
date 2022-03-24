#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  5 13:12:21 2021

@author: eilers
"""

import jax.numpy as jnp
import numpy as np
from jax import grad, value_and_grad, vjp, jit
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from jax.scipy.linalg import cho_solve, cho_factor
import scipy.optimize as op
import jax.scipy.optimize as jop
from numba import jit
import multiprocessing as mp

np.random.seed(31)

# -------------------------------------------------------------------------------
# data
# -------------------------------------------------------------------------------

def PCAInitial(X, Q):
    # impute data first if values are missing:
    if np.sum(np.isnan(X)) > 0:
       imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
       imp_mean.fit(X)
       X = imp_mean.transform(X)
    pca = PCA(Q)
    Z_initial = pca.fit_transform(X)
    return Z_initial


def Chi2_Matrix(Y1, Y1_ivar, Y2, Y2_ivar, infinite_diagonal = False):
    """
    Returns N1 x N2 matrix of chi-squared values.
    A clever user will decide on the Y1 vs Y2 choice sensibly.
    """
    N1, D = Y1.shape
    N2, D2 = Y2.shape
    assert D == D2
    assert Y1_ivar.shape == Y1.shape
    assert Y2_ivar.shape == Y2.shape
    chi2 = np.zeros((N1, N2)) 
#    # uses a loop to save memory
#    for n1 in range(N1):
#        DY = Y2 - Y1[n1, :]
#        denominator = Y2_ivar + Y1_ivar[n1, :]
#        ivar = Y2_ivar * Y1_ivar[n1, :] / (denominator + (denominator <= 0.))
#        chi2[n1, :] = np.dot(ivar * DY, DY.T)
    for n1 in range(N1):
        for n2 in range(N2):
            xx = Y2[n2, :] - Y1[n1, :]
            # asymmetric chi2 -- take only variance of testing object!
            denominator = Y2_ivar[n2, :] #+ Y1_ivar[n1, :]
            ivar = Y2_ivar[n2, :] / (denominator + (denominator <= 0.)) #* Y1_ivar[n1, :]
            chi2[n1, n2] = np.nansum(ivar * xx**2)
        if infinite_diagonal and n1 < N2:
            chi2[n1, n1] = np.Inf
    return chi2

def NN(index, chi2, labels):
    """
    Index is the second index into chi2!
    """
    N1 = labels.shape[0]
    foo, N2 = chi2.shape
    assert foo == N1
    assert index < N2
    return labels[np.argmin(chi2[:, index]), :], np.argmin(chi2[:, index])

# -------------------------------------------------------------------------------
# kernel
# -------------------------------------------------------------------------------

# radius basis function

def kernelRBF(Z, rbf, band): 
    B = B_matrix(Z)
    kernel = rbf * jnp.exp(band * B) 
    return kernel


def B_matrix(Z):
    return -0.5 * jnp.sum((Z[None, :, :] - Z[:, None, :]) ** 2, axis=2)

# -------------------------------------------------------------------------------
# optimization of latent variables
# -------------------------------------------------------------------------------

def cygnet_likelihood_d_worker_new(task):
    X, X_var, theta_rbf, Z, theta_band, good_stars = task
    kernel1 = kernelRBF(Z, theta_rbf, theta_band)    
    thiskernel = kernel1[good_stars, :][:, :, good_stars][0, :, 0, :] # kernel1[good_stars, :][:, good_stars]
    K1C = thiskernel + jnp.diag(X_var[good_stars])
    thisfactor = cho_factor(K1C, overwrite_a = True)
    thislogdet = 2. * jnp.sum(jnp.log(jnp.diag(thisfactor[0])))
    Lx = LxOrLy(thislogdet, thisfactor, X[good_stars])
    return Lx

def cygnet_likelihood_l_worker_new(task):
    Y, Y_var, gamma_rbf, Z, gamma_band, good_stars = task  
    kernel2 = kernelRBF(Z, gamma_rbf, gamma_band)
    thiskernel = kernel2[good_stars, :][:, :, good_stars][0, :, 0, :]
    K2C = thiskernel + jnp.diag(Y_var[good_stars])
    thisfactor = cho_factor(K2C, overwrite_a = True)
    thislogdet = 2. * jnp.sum(jnp.log(jnp.diag(thisfactor[0])))
    Ly = LxOrLy(thislogdet, thisfactor, Y[good_stars])
    return Ly


def LxOrLy(log_K_det, factor, data):  
    return -0.5 * log_K_det - 0.5 * jnp.dot(data, cho_solve(factor, data))

def lnL_Z_new(Z, hyper_params, theta_band, gamma_band, X, Y, X_var, Y_var, good_stars_X, good_stars_Y):    

    D = X.shape[1]
    L = Y.shape[1]
    Z = np.reshape(Z, (31, 16)) 
        
    theta_rbf, gamma_rbf = hyper_params

    #tasks = [(X[:, d], X_var[:, d], theta_rbf, Z, theta_band) for d in range(D)]
    Lx, Ly = jnp.array([0.]), jnp.array([0.])
    for d in range(D):
        task = X[:, d], X_var[:, d], theta_rbf, Z, theta_band, good_stars_X[d]
        Lx += cygnet_likelihood_d_worker_new(task)
    #for result in map(cygnet_likelihood_d_worker_new, tasks):
    #    Lx += result
     
    for l in range(L):
        task = Y[:, l], Y_var[:, l], gamma_rbf, Z, gamma_band, good_stars_Y[l]
        Ly += cygnet_likelihood_l_worker_new(task)
    
    #tasks = [(Y[:, l], Y_var[:, l], gamma_rbf, Z, gamma_band, good_stars_Y[l]) for l in range(L)]    
    #for result in map(cygnet_likelihood_l_worker_new, tasks):
    #    Ly += result
        
    Lz = -0.5 * jnp.sum(Z**2)
    L = Lx + Ly + Lz 
    print(-2.*L[0])
    return -2.*L[0]

# def over_lnL_Z_new(zs, hyper_params, theta_band, gamma_band, X, Y, Z_initial, X_var, Y_var):
#     l, gradl = value_and_grad(lnL_Z_new, 0)(zs, hyper_params, theta_band, gamma_band, X, Y, Z_initial, X_var, Y_var)
#     print(l, gradl)
#     return l, gradl
    
# -------------------------------------------------------------------------------
# optimization of hyper parameters
# -------------------------------------------------------------------------------

def lnL_h(pars, X, Y, Z, X_var, Y_var, X_mask = None, Y_mask = None):

    # hyper parameters shouldn't be negative
    if all(i >= 0 for i in pars):    
        D = X.shape[1]
        L = Y.shape[1]    

        theta_rbf, gamma_rbf = pars #theta_rbf0, 
        theta_band, gamma_band = 1., 1.
        
        kernel1 = kernelRBF(Z, theta_rbf, theta_band)
        kernel2 = kernelRBF(Z, gamma_rbf, gamma_band)
        
        if X_mask is None:
            X_mask = np.ones_like(X).astype(bool)
        if Y_mask is None:
            Y_mask = np.ones_like(Y).astype(bool)
        
        Lx, Ly = 0., 0.
        gradLx = np.zeros((1,)) # CHANGED THIS ON JULY 14 2021 (from 2, to 1,) for optimizing only the RBF parameters and not the BAND parameters
        gradLy = np.zeros((1,))
        for d in range(D):
            good_stars = X_mask[:, d]
            if np.sum(good_stars) > 0:
                #if d == 0:
                #    kernel1 = kernelRBF(Z, theta_rbf0, theta_band)
                thiskernel = kernel1[good_stars, :][:, good_stars]
                K1C = thiskernel + np.diag(X_var[good_stars, d])
                thisfactor = cho_factor(K1C, overwrite_a = True)
                thislogdet = 2. * np.sum(np.log(np.diag(thisfactor[0])))
                Lx += LxOrLy(thislogdet, thisfactor, X[good_stars, d]) 
                #if d == 0:
                #    gradLx0 = dLdhyper(X[good_stars, d], Z[good_stars, :], theta_band, theta_rbf0, thiskernel, thisfactor) 
                #else:
                gradLx += dLdhyper(X[good_stars, d], Z[good_stars, :], theta_band, theta_rbf, thiskernel, thisfactor) 
            
        for l in range(L):
            good_stars = Y_mask[:, l]
            #kernel2 = kernelRBF(Z, gamma_rbf[l], gamma_band)
            thiskernel = kernel2[good_stars, :][:, good_stars]
            K2C = thiskernel + np.diag(Y_var[good_stars, l])
            thisfactor = cho_factor(K2C, overwrite_a = True)
            thislogdet = 2. * np.sum(np.log(np.diag(thisfactor[0])))
            Ly += LxOrLy(thislogdet, thisfactor, Y[good_stars, l]) 
            gradLy += dLdhyper(Y[good_stars, l], Z[good_stars, :], gamma_band, gamma_rbf, thiskernel, thisfactor) 
         
        #Lz = - 0.5 * np.sum(Z**2)                      
        L = Lx + Ly #+ Lz
        gradL = np.hstack((gradLx, gradLy)) #gradLx0, 
        
        #print(-2.*Lx, -2.*Ly)
        #print(gradLx, gradLy)
        return -2.*L, -2.*gradL
    else:
        print('hyper parameters negative!') 
        return np.inf, np.inf * np.ones_like(pars) # hack! check again!

# -------------------------------------------------------------------------------
# likelihood derivatives
# -------------------------------------------------------------------------------


def dLdZ(data, Z, factor, K, band): 
    
    K_inv_data = cho_solve(factor, data)
    prefactor = band * K

    gradL = np.zeros_like(Z)
    N, Q = Z.shape # N might not be the same as global N, if labels have been dropped
    for n in range(N):
        #print l, Z.shape, (Z[good_stars, :] - Z[l, :]).shape, prefactor.shape
        lhat = np.zeros((N))
        lhat[n] = 1.
        vec = prefactor[:, n, None] * (Z - Z[n, :])
        gradL[n, :] += K_inv_data[n] * np.dot(K_inv_data, vec) - np.dot(cho_solve(factor, lhat), vec)
    
    return gradL


def dLdhyper(data, Z, band, rbf, K, factor):  
    
    K_inv_data = cho_solve(factor, data) 
    
    dKdrbf = 1./rbf * K  
    #B = B_matrix(Z)
    #dKdband = K * B    
    #dLdband = np.sum(0.5 * K_inv_data * np.dot(K_inv_data, dKdband)) - 0.5 * np.trace(cho_solve(factor, dKdband))
    dLdrbf = np.sum(0.5 * K_inv_data * np.dot(K_inv_data, dKdrbf)) - 0.5 * np.trace(cho_solve(factor, dKdrbf))
    return np.array([dLdrbf])#, dLdband])

# -------------------------------------------------------------------------------
# prediction
# -------------------------------------------------------------------------------


def mean_var(Z, Zj, data, data_var, rbf, band):
    N = Z.shape[0]
    B = np.zeros((N, ))
    for i in range(N):
        B[i] = -0.5 * jnp.dot((Z[i, :] - Zj).T, (Z[i, :] - Zj))   
    
    # prediction for test object: loop over d is in previous function
    if data.ndim == 1:
        K = kernelRBF(Z, rbf, band)
        KC = K + jnp.diag(data_var)
        k_Z_zj = rbf * jnp.exp(band * B)
        factor = cho_factor(KC, overwrite_a = True)
        mean = jnp.dot(data.T, cho_solve(factor, k_Z_zj))
        var = rbf - jnp.dot(k_Z_zj.T, cho_solve(factor, k_Z_zj))
        return mean, var, k_Z_zj, factor
    
    # prediction for training objects
    else:
        D = data.shape[1]
        mean_j = []
        var_j = [] 
        K = kernelRBF(Z, rbf, band)
        for d in range(D):
            #K = kernelRBF(Z, rbf[d], band)            
            KC = K + jnp.diag(data_var[:, d])
            #k_Z_zj = rbf[d] * np.exp(band * B)
            k_Z_zj = rbf * jnp.exp(band * B)
            factor = cho_factor(KC, overwrite_a = True)
            mean_j.append(jnp.dot(data[:, d].T, cho_solve(factor, k_Z_zj)))
            #var_j.append(rbf[d] - np.dot(k_Z_zj.T, cho_solve(factor, k_Z_zj)))
            var_j.append(rbf - jnp.dot(k_Z_zj.T, cho_solve(factor, k_Z_zj)))        
        return np.array(mean_j), np.array(var_j), k_Z_zj


def predictY(X_new, X_var_new, X, X_var, Y, Y_var, Z_final, hyper_params, y0, z0, X_mask = None, Y_mask = None, X_mask_new = None):

    theta_rbf, theta_band, gamma_rbf, gamma_band = hyper_params
            
    res = op.minimize(lnL_znew, x0 = z0, args = (X_new, X_var_new, Z_final, X, X_var, theta_rbf, theta_band, X_mask, X_mask_new), method = 'L-BFGS-B', jac = True, 
                   options={'gtol':1e-12, 'ftol':1e-12})   
    Z_new = res.x
    success_z = res.success
    print('latent variable optimization - success: {}'.format(res.success))
    
    res = op.minimize(lnL_ynew, x0 = y0, args = (Z_new, Z_final, Y, Y_var, gamma_rbf, gamma_band, Y_mask), method = 'L-BFGS-B', jac = True, 
                   options={'gtol':1e-12, 'ftol':1e-12})   
    Y_new = res.x
    success_y = res.success
    print('new labels optimization - success: {}'.format(res.success)) 
    
    return Z_new, Y_new, success_z, success_y

def lnL_znew(pars, X_new_j, X_var_new_j, Z, X, X_var, rbf, band, X_mask = None, X_mask_new = None): 
    
    if X_mask is None:
        X_mask = np.ones_like(X).astype(bool)
    if X_mask_new is None:
        X_mask_new = np.ones_like(X_new_j).astype(bool) 
        
    Zj = pars
    D = X_new_j.shape[0]  
    
    like = 0.
    gradL = 0.
    for d in range(D):
        if X_mask_new[d] == True:
            good_stars = X_mask[:, d]
            mean, var, k_Z_zj, factor = mean_var(Z[good_stars, :], Zj, X[good_stars, d], X_var[good_stars, d], rbf, band) # used to be rbf[d]
            assert var > 0.
            
            like += -0.5 * np.dot((X_new_j[d] - mean).T, (X_new_j[d] - mean)) / \
                              (var + X_var_new_j[d]) - 0.5 * np.log(var + X_var_new_j[d]) 
            
            dLdmu, dLdsigma2 = dLdmusigma2(X_new_j[d], mean, (var + X_var_new_j[d]))
            dmudZ, dsigma2dZ = dmusigma2dZ(X[good_stars, d], factor, Z[good_stars, :], Zj, k_Z_zj, band)        
            gradL += np.dot(dLdmu, dmudZ) + np.dot(dLdsigma2, dsigma2dZ)    
    return -2.*like, -2.*gradL

def lnL_ynew(pars, Zj, Z, Y, Y_var, rbf, band, Y_mask = None):  
    
    if Y_mask is None:
        Y_mask = np.ones_like(Y).astype(bool)
    
    lj = pars
    L = Y.shape[1]
    
    like = 0.
    gradL = []
    for l in range(L):
        good_stars = Y_mask[:, l]
        mean, var, k_Z_zj, factor = mean_var(Z, Zj, Y[good_stars, l], Y_var[good_stars, l], rbf, band) # used to be rbf[l]
        assert var > 0.
        like += -0.5 * np.dot((lj[l] - mean).T, (lj[l] - mean)) / var - 0.5 * np.log(var)         
        gradL.append( -(lj[l] - mean) / var )    
    return -2.*like, -2.*np.array(gradL)


def predictX(Y_new, X, X_var, Y, Y_var, Z_final, hyper_params, x0, z0, X_mask = None, Y_mask = None):
    
    theta_rbf, theta_band, gamma_rbf, gamma_band = hyper_params

    res = op.minimize(lnL_znew_givenY, x0 = z0, args = (Y_new, Z_final, Y, Y_var, gamma_rbf, gamma_band, Y_mask), method = 'L-BFGS-B', #jac = True, 
                   options={'gtol':1e-12, 'ftol':1e-12})   
    Z_new = res.x
    print('latent variable optimization - success: {}'.format(res.success)) 
    
    res = op.minimize(lnL_xnew_givenY, x0 = x0, args = (Z_new, Z_final, X, X_var, theta_rbf, theta_band, X_mask), method = 'L-BFGS-B', jac = True, 
                   options={'gtol':1e-12, 'ftol':1e-12})   
    X_new = res.x
    print('new labels optimization - success: {}'.format(res.success)) 
    
    return Z_new, X_new
    
def lnL_znew_givenY(pars, Y_new, Z, Y, Y_var, rbf, band, Y_mask = None): 
    
    if Y_mask is None:
        Y_mask = np.ones_like(Y).astype(bool)
        
    Zj = pars
    L = Y_new.shape[0]  
    
    like = 0.
    #gradL = 0.
    for l in range(L):
        good_stars = Y_mask[:, l]
        mean, var, k_Z_zj, factor = mean_var(Z[good_stars, :], Zj, Y[good_stars, l], Y_var[good_stars, l], rbf, band) # used to be rbf[d]
        assert var > 0.        
        like += -0.5 * np.dot((Y_new[l] - mean).T, (Y_new[l] - mean)) / var      
        #dLdmu, dLdsigma2 = dLdmusigma2(Y_new[l], mean, var)
        #dmudZ, dsigma2dZ = dmusigma2dZ(Y[good_stars, l], factor, Z[good_stars, :], Zj, k_Z_zj, band)        
        #gradL += np.dot(dLdmu, dmudZ) + np.dot(dLdsigma2, dsigma2dZ)  
    return -2.*like#, -2.*gradL


def lnL_xnew_givenY(pars, Zj, Z, X, X_var, rbf, band, X_mask = None):  
    
    if X_mask is None:
        X_mask = np.ones_like(X).astype(bool)
    
    lj = pars
    D = X.shape[1]
    
    like = 0.
    gradL = np.zeros((D))
    for d in range(D):
        good_stars = X_mask[:, d]
        mean, var, k_Z_zj, factor = mean_var(Z[good_stars, :], Zj, X[good_stars, d], X_var[good_stars, d], rbf, band) # used to be rbf[l]
        assert var > 0.
        like += -0.5 * np.dot((lj[d] - mean).T, (lj[d] - mean)) / var - 0.5 * np.log(var)         
        gradL[d] = ( -(lj[d] - mean) / var )
    return -2.*like, -2.*gradL


def dLdmusigma2(X_new_j, mean, var):
    dLdmu = (X_new_j - mean) / var
    dLdsigma2 = -0.5 / var + 0.5 * np.dot((X_new_j - mean).T, (X_new_j - mean)) / var**2.
    return dLdmu, dLdsigma2


def dmusigma2dZ(data, factor, Z, Zj, k_Z_zj, band):
    term2 = k_Z_zj[:, None] * band * (Z - Zj)
    Kinv_term2 = cho_solve(factor, term2)
    dmudZ = np.dot(data.T, Kinv_term2)    
    
    dsigma2dZ = -2. * np.dot(k_Z_zj , Kinv_term2)    
    return dmudZ, dsigma2dZ



# -------------------------------------------------------------------------------
# new versin of lnLZ
# -------------------------------------------------------------------------------

def cygnet_likelihood_d_worker(task):
    obj, d = task
    good_stars = obj.X_mask[:, d]
    thiskernel = obj.kernel1[good_stars, :][:, good_stars]
    K1C = thiskernel + np.diag(obj.X_var[good_stars, d])
    thisfactor = cho_factor(K1C, overwrite_a = True)
    thislogdet = 2. * np.sum(np.log(np.diag(thisfactor[0])))
    Lx = LxOrLy(thislogdet, thisfactor, obj.X[good_stars, d])
    gradLx = np.zeros_like(obj.Z)
    gradLx[good_stars, :] = dLdZ(obj.X[good_stars, d], obj.Z[good_stars, :], thisfactor, thiskernel, obj.theta_band)        
    return Lx, gradLx

def cygnet_likelihood_l_worker(task):
    obj, l = task    
    good_stars = obj.Y_mask[:, l]
    thiskernel = obj.kernel2[good_stars, :][:, good_stars]
    K2C = thiskernel + np.diag(obj.Y_var[good_stars, l])
    thisfactor = cho_factor(K2C, overwrite_a = True)
    thislogdet = 2. * np.sum(np.log(np.diag(thisfactor[0])))
    Ly = LxOrLy(thislogdet, thisfactor, obj.Y[good_stars, l])
    gradLy = np.zeros_like(obj.Z)
    gradLy[good_stars, :] = dLdZ(obj.X[good_stars, l], obj.Z[good_stars, :], thisfactor, thiskernel, obj.gamma_band)        
    return Ly, gradLy

class CygnetLikelihood():
    
    def __init__(self, X, Y, Q, hyper_params, X_var, Y_var, X_mask = None, Y_mask = None):
        '''
        X: pixel data
        Y: label data
        '''
        # change X and Y!
        self.X = X.copy()
        self.Y = Y.copy()
        self.X_var = X_var.copy()
        self.Y_var = Y_var.copy()
        assert self.X.shape == self.X_var.shape
        assert self.Y.shape == self.Y_var.shape        
        self.N, self.D = self.X.shape
        N, self.L = self.Y.shape
        assert N == self.N
        self.Q = Q
        self.theta_rbf, self.theta_band, self.gamma_rbf, self.gamma_band = hyper_params
        if X_mask is None:
            self.X_mask = np.ones_like(X).astype(bool)
        else:
            self.X_mask = X_mask.copy()
        if Y_mask is None:
            self.Y_mask = np.ones_like(Y).astype(bool)    
        else:
            self.Y_mask = Y_mask.copy()
        
        # container to hold the summed likelihood value, gradients
        self._L = 0.
        self._gradL = np.zeros_like(self.Z)
        
    def update_pars(self, pars):
        
        self.Z = np.reshape(pars, (self.N, self.Q))
        self.kernel1 = kernelRBF(self.Z, self.theta_rbf, self.theta_band)
        self.kernel2 = kernelRBF(self.Z, self.gamma_rbf, self.gamma_band)

    def __call__(self, pars, pool):
        self.update_pars(pars)        
        
        tasks = [(self, d) for d in range(self.D)]
        Lx = 0.
        gradLx = np.zeros_like(self.Z)
        for result in pool.map(cygnet_likelihood_d_worker, tasks):
            Lx += result[0]
            gradLx += result[1]
        
        # reset containers
        cygnet_likelihood._L = 0.    
        cygnet_likelihood._gradL = np.zeros_like(cygnet_likelihood.Z)
        for _ in pool.map(cygnet_likelihood, zip(range(L), ['l']*L)):
            pass
        Ly = cygnet_likelihood._L
        gradLy = cygnet_likelihood._gradL
        
        Lz = -0.5 * np.sum(cygnet_likelihood.Z**2)
        dlnpdZ = -cygnet_likelihood.Z 
        L = Lx + Ly + Lz   
        gradL = gradLx + gradLy + dlnpdZ      
        gradL = np.reshape(gradL, (cygnet_likelihood.N * cygnet_likelihood.Q, ))   # reshape gradL back into 1D array   
        print(-2.*Lx, -2.*Ly, -2.*Lz)
        
        return -2.*L, -2.*gradL
                
        if l_or_d == 'd':
            return self.lnLd(index)
        elif l_or_d == 'l':
            return self.lnLl(index)
    
    def lnLl(self, l):
        good_stars = self.Y_mask[:, l]
        thiskernel = self.kernel2[good_stars, :][:, good_stars]
        K2C = thiskernel + np.diag(self.Y_var[good_stars, l])
        thisfactor = cho_factor(K2C, overwrite_a = True)
        thislogdet = 2. * np.sum(np.log(np.diag(thisfactor[0])))
        Ly = LxOrLy(thislogdet, thisfactor, self.Y[good_stars, l])
        gradLy = np.zeros_like(self.Z)
        gradLy[good_stars, :] = dLdZ(self.X[good_stars, l], self.Z[good_stars, :], thisfactor, thiskernel, self.gamma_band)        
        return Ly, gradLy
    
    def callback(self, result):
        _L, _gradL = result
        self._L += _L
        self._gradL += _gradL

def lnL_Z(pars, X, Y, Q, hyper_params, pool, X_var, Y_var, X_mask = None, Y_mask = None):  
        
    cygnet_likelihood = CygnetLikelihood(pars, X, Y, Q, hyper_params, X_var, Y_var, X_mask, Y_mask)
    D = cygnet_likelihood.D    
    L = cygnet_likelihood.L    
    
    for _ in pool.map(cygnet_likelihood, zip(range(D), ['d']*D)):
        pass
    Lx = cygnet_likelihood._L
    gradLx = cygnet_likelihood._gradL
    
    # reset containers
    cygnet_likelihood._L = 0.    
    cygnet_likelihood._gradL = np.zeros_like(cygnet_likelihood.Z)
    for _ in pool.map(cygnet_likelihood, zip(range(L), ['l']*L)):
        pass
    Ly = cygnet_likelihood._L
    gradLy = cygnet_likelihood._gradL
    
    Lz = -0.5 * np.sum(cygnet_likelihood.Z**2)
    dlnpdZ = -cygnet_likelihood.Z 
    L = Lx + Ly + Lz   
    gradL = gradLx + gradLy + dlnpdZ      
    gradL = np.reshape(gradL, (cygnet_likelihood.N * cygnet_likelihood.Q, ))   # reshape gradL back into 1D array   
    print(-2.*Lx, -2.*Ly, -2.*Lz)
    
    return -2.*L, -2.*gradL


