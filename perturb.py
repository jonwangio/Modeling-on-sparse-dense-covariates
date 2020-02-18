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


# Generate dense covariate as noisy versions of the ground truth GP
def noiseCov(X, Y, mean, std):
    # Simple noise follows N(mean, std**2)
    Ynoise = std*np.random.randn(Y.size, 1)+mean
    Y += Ynoise  # Call surface function
    gt.showGrid(X, Y)
    return (X, Y)
    
# More advanced noise from the "Colors of Noise"
def noiseBasis():
    
    return None

def noiseColorCov():
    
    return None
    

# Generate dense covariate with advanced GP controlled noise
def noiseGPCov(X, Y, var, gamma):
    # GP controlled noise drawn from a covariance matrix generated from kernel
    K = gt.arbKer(X, X, var=1, gamma=1)  # Reuse the arbitrary kernel to control noise
    # Cholesky decomposition
    L = np.linalg.cholesky(K + 1e-6*np.eye(K.shape[0]))
    # Draw sample (can be multiple)
    Y_prior = np.dot(L, np.random.normal(size=(X.shape[0],1)))  # Draw 1 sample
    Y += Y_prior
    gt.showGrid(X, Y)
    gt.showGrid(X, Y_prior)
    return(X, Y_prior)
