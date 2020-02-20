#-------------------------------------------------------------------------------
# Name:        Learning spatial processes through sparse measurements 
#              and its dense gridded covariate(s)

##########
# Perturbation functions
##########


# Purpose:     
#              1. Aiming at the general problem of populating sparse in-situs with EO data
#              2. Starting from artificially created dummy data
#              3. Deducting the problems to special study cases
#              Main method is ...
#
# Modifications ~~:
#              1.
#
# Author:      Jiong Wang
#
# Created:     08/01/2020
# Copyright:   (c) JonWang 2020
# Licence:     <your licence>
#-------------------------------------------------------------------------------

import numpy as np
import groundTruth as gt
import point as pt
import covar as cov
import perturb as pb

from scipy import interpolate

#####################################################
# 01 Generate dense covariate as noisy versions of the ground truth GP
#####################################################

# Simple random/white noise controlled by variance and sparsity
def noiseCov(X, Y, var, spar):
    # Grid same to the original X
    xg1 = np.linspace(X[:,0].min(), X[:,0].max(), int(np.sqrt(len(X))))
    xg2 = np.linspace(X[:,1].min(), X[:,1].max(), int(np.sqrt(len(X))))
    grid_x, grid_y = np.meshgrid(xg1, xg2)
    
    # Grid scaled by sparsity parameter to create random/white noise
    pix_size = np.ptp(X[:,0])/np.sqrt(len(X))  # Original pixel size
    s = spar/pix_size  # Scale factor as number of pixels
    xxg1 = np.linspace(X[:,0].min(), X[:,0].max(), int(np.sqrt(len(X))/s))
    xxg2 = np.linspace(X[:,1].min(), X[:,1].max(), int(np.sqrt(len(X))/s))
    grid_xx, grid_yy = np.meshgrid(xxg1, xxg2)
    
    ind = np.array([grid_xx.ravel(), grid_yy.ravel()]).T
    Ynoise = np.sqrt(var)*np.random.randn(ind.shape[0], 1)
    Ynoise = interpolate.griddata(ind, Ynoise, (grid_x, grid_y), method='linear')
    Ynoise = Ynoise.reshape(-1,1)
    Y += Ynoise  # Call surface function
    gt.showGrid(X, Y)
    return (X, Y)
    
# More advanced noise from the "Colors of Noise"

def noiseColorCov():
    
    return None
    

# Generate dense covariate with advanced GP controlled noise
def noiseGPCov(X, Y, var, gamma):
    if var==0 or gamma==0:
        Y_noise = Y
    else:
        # GP controlled noise drawn from a covariance matrix generated from kernel
        K = gt.arbKer(X, X, var, gamma)  # Reuse the arbitrary kernel to control noise
        # Cholesky decomposition
        L = np.linalg.cholesky(K + 1e-6*np.eye(K.shape[0]))
        # Draw sample (can be multiple)
        Y_noise = np.dot(L, np.random.normal(size=(X.shape[0],1)))  # Draw 1 sample
    Ynew = Y+Y_noise
    gt.showGrid(X, Ynew)
    gt.showGrid(X, Y_noise)
    return(X, Ynew)
