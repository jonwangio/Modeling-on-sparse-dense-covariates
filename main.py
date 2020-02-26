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

#==================================
# 01_1 1D single process example
#==================================

# Experimental taste of the Gaussian Process (GP) regression
# Sample data from a linearly transformed sinusoidal function
X = np.random.rand(30)[:,None]; X=X*15-5
Y = np.sin(X) + .2*X + np.random.randn(len(X),1)*0.05
pl.figure(figsize=(12,4))
pl.plot(X,Y,'kx',mew=1.5)
plt.xlim([-6., 11])
plt.ylim([-4., 3.5])

# Specify the kernel and model
k = GPy.kern.RBF(input_dim=1, variance=1., lengthscale=1.)
m = GPy.models.GPRegression(X,Y,k)
m.plot()

# Manually optimize the model and show trained parameters
m['.*var']=2.
m['.*leng']=0.5
m['Gaussian_noise.variance']=0.02
m.plot()
print(m)

# Automatically optimize the model and show trained parameters
m.optimize()
m.plot()
print(m)

# Posterior realizations of the optimized model
testX = np.linspace(-5., 15., 1000).reshape(-1, 1)
posteriorTestY = m.posterior_samples_f(testX, full_cov=True, size=3).reshape(-1,3) # Draw 3 realizations
simY, simMse = m.predict(testX)

pl.figure(figsize=(12,4))
plt.plot(testX, posteriorTestY, linewidth=.5)
plt.plot(X, Y, 'kx', markersize=4)
plt.plot(testX, simY - 3 * simMse ** 0.5, '--g', linewidth=.5)
plt.plot(testX, simY + 3 * simMse ** 0.5, '--g', linewidth=.5)


#==================================
# 01_2 1D coregionalized Gaussian Process
#==================================

# Experimental taste of 1D coregionalized Gaussian Process (GP) regression
# Sample data from exactly the same sinusoidal function as above
X1 = np.random.rand(10)[:,None]; X1=X1*12-5  # Process 1
X1t = np.random.rand(30)[:,None]; X1t=X1t*15-5  # Process 1 test
X2 = np.random.rand(8)[:,None]; X2=X2*6+4  # Process 2
X2t = np.random.rand(30)[:,None]; X2t=X2t*15-5  # Process 2 test

Y1 = np.sin(X1) + .3*X1 + np.random.randn(len(X1),1)*0.05  # From same function as above
Y1t = np.sin(X1t) + .3*X1t + np.random.randn(len(X1t),1)*0.05  # Y1 Test data
Y2 = np.sin(X2) - .2*X2 + np.random.randn(len(X2),1)*0.05
Y2t = np.sin(X2t) - .2*X2t + np.random.randn(len(X2t),1)*0.05  # Y2 Test data

# Plot observations
fig = pl.figure(figsize=(12,8))
# Process 1
ax1 = fig.add_subplot(211)
ax1.set_xlim([-6., 11])
ax1.set_ylim([-4, 3.5])
ax1.set_title('Process 1')
ax1.plot(X1,Y1,'kx', X1t,Y1t,'rx')
#Output 2
ax2 = fig.add_subplot(212)
ax1.set_xlim([-6., 11])
ax1.set_ylim([-4, 3.5])
ax2.set_title('Process 2')
ax2.plot(X2,Y2,'ko', X2t,Y2t,'ro')


# Coregionalized Gaussian Process
K = GPy.kern.RBF(1)
icm = GPy.util.multioutput.ICM(input_dim=1,num_outputs=2,kernel=K)
m2 = GPy.models.GPCoregionalizedRegression([X1,X2],[Y1,Y2],kernel=icm)
m2.optimize()

# Plot
def plot_2outputs(m,xlim,ylim):
    fig = pl.figure(figsize=(12,8))
    #Output 1
    ax1 = fig.add_subplot(211)
    ax1.set_xlim(xlim)
    ax1.set_title('Output 1')
    m.plot(plot_limits=xlim,fixed_inputs=[(1,0)],which_data_rows=slice(0,len(X1)),ax=ax1)
    ax1.plot(X1t[:,:1],Y1t,'rx',mew=1.5)
    #Output 2
    ax2 = fig.add_subplot(212)
    ax2.set_xlim(xlim)
    ax2.set_title('Output 2')
    m.plot(plot_limits=xlim,fixed_inputs=[(1,1)],which_data_rows=slice(len(X1),len(X1)+len(X2)),ax=ax2)
    ax2.plot(X2t[:,:1],Y2t,'rx',mew=1.5)

plot_2outputs(m2, xlim = (-6., 11), ylim = (-2., 2.))
 
 
#####################################################
# 02 2D Gaussian Process
#####################################################
   
#==================================
# 02_1 Ground truth data generation
#==================================
# Dummy gridded dataset over input space
x1min, x1max, x2min, x2max = -5, 10, 0, 15  # Domain of the input space
row, col = 30, 30  # Grid number of the input space
var, gamma = 15**2, 3
X, Y = gt.grid(x1min,x1max,x2min,x2max,row,col,var,gamma)  # Dummy grid dataset realized by function
#X, Y = np.load('X.npy'), np.load('Y.npy')
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
        r = 2
        
        # X, Y = noiseCov(Xp, Yp, mean=100, std=30)  # Add noise to GP ground truth before point sampling
        x, y = pt.poissonPt(Xp, Yp, r)
        
        # Dense covariate(s) with noise
        Xcov, Ycov = cov.linCov(Xp, Yp)  # Dense covariate through linear transformation
        '''
        !!! Dense covariate with controlled noise Xn, Yn !!!
        '''
        sc_1, sc_2 = (1/15)*s1, (1/15)*s2  # Each scenario runs with scaled var and gamma
        # White/random noise
        Xn, Yn = pb.noiseCov(Xcov, Ycov, var=(sc_1*np.sqrt(var*0.2))**2, spar=sc_2*gamma)
        # GP controlled smooth noise
        #Xn, Yn = pb.noiseGPCov(Xcov, Ycov, var=(sc_1*np.sqrt(var))**2, gamma=sc_2*gamma)
        
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

# Visualizing the results
err = np.load('RMSE_all.npy')
err = np.array(err)
err = err[~np.any(err == 0, axis=1)]  # Remove rows with 0
#err = err[~np.any(err > 50, axis=1)]  # Remove the extrema

## In 2D
plt.scatter(err[:,0], err[:,1], c=err[:,2], cmap='coolwarm', s=15)
plt.xlabel('Noise intensity over information')
plt.ylabel('Noise lengthscale over information')

# In 3D
fig = plt.figure()
ax = plt.axes(projection='3d')
#ax.scatter(err[:,0], err[:,1], err[:,2]/15, c='coolwarm', marker='o', s=3)
ax.plot_trisurf(err[:,0], err[:,1], err[:,2]/15, cmap='coolwarm', edgecolor='none')  # Error over info. std.
ax.set_xlabel('Noise intensity over information')
ax.set_ylabel('Noise lengthscale over information')
ax.set_zlabel('Error over information')

#####################################################
# 03 REAL Training, testing and validation
#####################################################




