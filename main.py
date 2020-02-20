#-------------------------------------------------------------------------------
# Name:        Learning spatial processes through sparse measurements 
#              and its dense gridded covariate(s)

##########
# Main
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
import os
import matplotlib.pyplot as plt
import pylab as pl
import GPy

'''
from scipy import stats
from random import random
from math import cos, sin, floor, sqrt, pi, ceil
from mpl_toolkits.mplot3d import Axes3D
'''

import groundTruth as gt
import point as pt
import covar as cov
import perturb as pb


#####################################################
# 01 1D Toy Gaussian Process
#####################################################

# Experimental taste of the Gaussian Process (GP) regression
# Sample data from a sin/cos conjunctions
X = np.linspace(0.05,0.95,10)[:,None]
Y = -np.cos(np.pi*X) +np.sin(4*np.pi*X) + np.random.randn(10,1)*0.05
pl.figure()
pl.plot(X,Y,'kx',mew=1.5)

# Specify the kernel and model
k = GPy.kern.RBF(input_dim=1, variance=1., lengthscale=1.)
m = GPy.models.GPRegression(X,Y,k)
m.plot()

# Manually optimize the model and show trained parameters
m['.*var']=2.
m['.*leng']=0.5
m['Gaussian_noise.variance']=0.001
m.plot()
print(m)

# Automatically optimize the model and show trained parameters
m.optimize()
m.plot()
print(m)

# Posterior realizations of the optimized model
testX = np.linspace(0, 1, 101).reshape(-1, 1)
posteriorTestY = m.posterior_samples_f(testX, full_cov=True, size=3).reshape(-1,3) # Draw 3 realizations
simY, simMse = m.predict(testX)

plt.plot(testX, posteriorTestY, linewidth=.5)
plt.plot(X, Y, 'ok', markersize=4)
plt.plot(testX, simY - 3 * simMse ** 0.5, '--g', linewidth=.5)
plt.plot(testX, simY + 3 * simMse ** 0.5, '--g', linewidth=.5)

 
#####################################################
# 02 2D Gaussian Process
#####################################################
   
#==================================
# 02_1 Ground truth data generation
#==================================
# Dummy gridded dataset over input space
x1min, x1max, x2min, x2max = -5, 10, 0, 15  # Domain of the input space
row, col = 40, 40  # Grid number of the input space
var, gamma = 15**2, 3
X, Y = gt.grid(x1min,x1max,x2min,x2max,row,col,var,gamma)  # Dummy grid dataset realized by function
gt.showGrid(X, Y)  # Show function dummy grid

# Ground truth representation through GP (parameters are var and lengthscale)
scale = 1   # Define GP prediction/approximation space as densified input space
Xp, Yp, m = gt.gtGP(X, Y, scale)  # Approximation
gt.showGrid(Xp, Yp)  # Dummy grid dataset approximated by GP as ground truth

#==================================
# 02_2 Scenario test
#==================================
# BASE SCENARIO: blind point samples and noise-free linear transformed covariate
scen_1 = 30  # Total scenarios as number of parameter values
scen_2 = 30
#totalScen = f(lengthscale)
corr = []  # Inferred coregionalized GP correlation
lengthscales = []  # Inferred lengthscale
Y_hatAll = []  # Inferred Y_hat
RMSE_all = []  # Mean sqaured error between Yp and Y_hat
np.save('RMSE_all.npy', RMSE_all)

for s1 in range(scen_1):
    for s2 in range(scen_2):
        # Point samples as point observation
        #r = s+1  # Minimal distance (number of grid) separating points
        r = 3
        
        # X, Y = noiseCov(Xp, Yp, mean=100, std=30)  # Add noise to GP ground truth before point sampling
        x, y = pt.poissonPt(Xp, Yp, r)
        
        # Dense covariate(s) with noise
        Xcov, Ycov = cov.linCov(Xp, Yp)  # Dense covariate through linear transformation
        '''
        !!! Dense covariate with controlled noise Xn, Yn !!!
        '''
        #Xn, Yn = noiseCov(Xcov, Ycov, mean=-0, std=30)
        sc_1, sc_2 = (1/15)*s1, (1/15)*s2  # Each scenario runs with scaled var and gamma
        Xn, Yn = pb.noiseGPCov(Xcov, Ycov, var=(sc_1*np.sqrt(var))**2, gamma=sc_2*gamma)
        
        # Model inference through GP Coregionalization
        mCov, Bnorm = gt.coregionGP(x, y, Xn, Yn)  # Coregionalization model
        
        # Prediction through optimized model
        Xnew = np.hstack([Xp,np.zeros_like(Yp)])  # Using existing Xp as new location for prediction on sparse process
        noise_dict = {'output_index':Xnew[:,-1].astype(int)}  # Indicate noise model to be used
        Y_hat = mCov.predict(Xnew,Y_metadata=noise_dict)[0]
        
        RMSE = np.sqrt(np.square(np.subtract(Yp,Y_hat)).mean())
        
        corr.append(Bnorm[0,1])
        lengthscales.append(mCov.ICM.rbf.lengthscale)
        Y_hatAll.append(Y_hat)
        
        RMSE = np.array([sc_1, sc_2, RMSE])
        RMSE_all.append(RMSE)
        
        #showGrid(Xnew[:,:-1], Y_hat)
        #xinf = mCov.X[len(x):,0:-1]
        #yinf = mCov.Y[len(y):]
        #showGrid(xinf, yinf)
        
        plt.close('all')
        print("Finished scenario: ", s1, s2)
        print("RMSE is: ", RMSE)
        print("Lengthscale is: ", mCov.ICM.rbf.lengthscale)
        
        os.remove('RMSE_all.npy')
        np.save('RMSE_all.npy', RMSE_all)

RMSE_all = np.array(RMSE_all)
plt.scatter(RMSE_all[:,0], RMSE_all[:,1], c=RMSE_all[:,2], s=30)



#####################################################
# 03 REAL Training, testing and validation
#####################################################




