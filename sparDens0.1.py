#-------------------------------------------------------------------------------
# Name:        Learning spatial processes through sparse measurements 
#              and its dense gridded covariate(s)

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
import matplotlib.pyplot as plt
import pylab as pb
import GPy

from scipy import stats
from mpl_toolkits.mplot3d import Axes3D  


#####################################################
# 01 Toy Gaussian Process in 1-dimensional
#####################################################

# Experimental taste of the Gaussian Process (GP) regression
# Sample data from a sin/cos conjunctions
X = np.linspace(0.05,0.95,10)[:,None]
Y = -np.cos(np.pi*X) +np.sin(4*np.pi*X) + np.random.randn(10,1)*0.05
pb.figure()
pb.plot(X,Y,'kx',mew=1.5)

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
# 02 Create dummy sparse and dense datasets
#####################################################

# Grid dummy dataset with specified function over input space with rows (r) and columns (c).
# The grid dataset is then approximated by 2D (combined) Gaussian Process as a ground truth process.
# The ground truth process is then used to test and validate different models.

#==================================
# 02_1 Real function based dummy dataset
#==================================
# Function options for creating dummy dataset over input space
# Branin function
def branin(X):
    y = (X[:,1]-5.1/(4*np.pi**2)*X[:,0]**2+5*X[:,0]/np.pi-6)**2
    y += 10*(1-1/(8*np.pi))*np.cos(X[:,0])+10
    return(y)

# ... function
# ...
# ...

# Dummy grid dataset defined over input space
def grid(r, c):
    xg1 = np.linspace(-5,10,r)
    xg2 = np.linspace(0,15,c)

    X = np.zeros((xg1.size * xg2.size,2))
    for i,x1 in enumerate(xg1):
        for j,x2 in enumerate(xg2):
            X[i+xg1.size*j,:] = [x1,x2]
    Y = branin(X)  # Call surface function
    return(X, Y[:,None])

# Show dummy grid dataset
def showGrid(X, Y, r=5, c=5):
    x1, x2 = X.reshape(r,c,2)[:,:,0], X.reshape(r,c,2)[:,:,1]
    Y = Y.reshape(r,c)
    print(x1.shape, x2.shape, Y.shape)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim3d(np.min(x1),np.max(x1))
    ax.set_ylim3d(np.min(x2),np.max(x2))
    ax.set_zlim3d(np.min(Y),np.max(Y))
    ax.plot_surface(x1, x2, Y)
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('Y')
    plt.show()
    return None

#==================================
# 02_2 GP realization as ground truth processes
#==================================
# Realization of dummy grid data through GP
# Kernel options
def kGP():
    kg = GPy.kern.RBF(input_dim=2, ARD = True)
    kb = GPy.kern.Bias(input_dim=2)
    k = kg + kb
    k.plot()
    return(k)

# Model specification
def mGP(X, Y):
    k = kGP()  # Kernel
    
    m = GPy.models.GPRegression(X,Y,k,normalizer=True)  # Specify model
    m.sum.bias.variance.constrain_bounded(1e-3,1e5)
    m.sum.rbf.variance.constrain_bounded(1e-3,1e5)
    m.sum.rbf.lengthscale.constrain_bounded(.1,200.)
    m.Gaussian_noise.variance.constrain_fixed(1e-3, 1e-1)
    
    m.randomize()  # Random initialization
    m.optimize()  # Optimization
    m.plot()
    return(m)

# GP realization of dummy function-based dataset as ground truth GP
def gtGP(X, Y, r, c):
    m = mGP(X, Y)  # Train the model over dummy grid dataset
    
    xg1 = np.linspace(-5,10,r)
    xg2 = np.linspace(0,15,c)

    Xp = np.zeros((xg1.size * xg2.size,2))
    for i,x1 in enumerate(xg1):
        for j,x2 in enumerate(xg2):
            Xp[i+xg1.size*j,:] = [x1,x2]
    # Draw dense GP approximation to the dummy through trained model
    Yp = m.predict(Xp)[0]
    return(Xp, Yp)
    
# Uncertainty/accuracy control of the GP realization: ground truth GP needs to be GOOD!
def accGP():
    

    
#==================================
# 02_3 Sparse point observation from the ground truth processes
#==================================
# TWO OPTIONS!!!
# 1_perturbate the ground truth GP before sampling
# 2_perturbate the randomly sampled points from the GP
# In either way, we need two types of functions: 
# (1) random points generation functions, and (2) perturbation functions
    
# Random point sample from the grid surface
def randPt(X,Y,n):
    dim = X.shape[0]  # Input data dimension
    randInd = np.random.randint(0,dim,size=n)  # Index of n random draw
    
    x = X[randInd,:]  # Random draw of X
    y = Y[randInd,:]  # Random draw of Y
    
    # Plot 3d random points
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.set_xlim3d(np.min(x[:,0]),np.max(x[:,0]))
    ax.set_ylim3d(np.min(x[:,1]),np.max(x[:,1]))
    ax.set_zlim3d(np.min(Y),np.max(Y))
    ax.scatter(x[:,0], x[:,1], y)
    plt.show()
    return (x, y)

# More strategic sampling from characteristic locations (critical points?)
def cPt(X,Y,n):
    Z = Y.reshape(r,c)  # Reshape to surface
    Zy, Zx = np.gradient(Z)  # Get local minima and maxima through gradient
    Zy, Zx = Zy.ravel(), Zx.ravel()  # Reshape into single column                                            
    ind = np.where(((Zx>=-0.5) & (Zy<=0.5)) | ((Zy>=-0.5) & (Zy<=0.5)))  # Index of critical points
    randInd = np.random.choice(ind[0], n)
    
    x = X[randInd,:]  # Random draw of X
    y = Y[randInd,:]  # Random draw of Y
    
    # Plot 3d random points
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.set_xlim3d(np.min(x[:,0]),np.max(x[:,0]))
    ax.set_ylim3d(np.min(x[:,1]),np.max(x[:,1]))
    ax.set_zlim3d(np.min(Y),np.max(Y))
    ax.scatter(x[:,0], x[:,1], y)
    plt.show()
    return (x, y)

        
 
#==================================
# 02_4 Dense covariate from the ground truth processes
#==================================








#==================================
# 02_5 Section '__main__'
#==================================
# Dummy gridded dataset over input space
r, c = 100, 100  # Define input space 
X, Y = grid(r,c)  # Dummy grid dataset realized by function
showGrid(X, Y, r, c)  # Show function dummy grid

rGP, cGP = 100, 100   # Define GP prediction/approximation space over input space
Xp, Yp = gtGP(X, Y, rGP, cGP)  # Approximation
showGrid(Xp, Yp, rGP, cGP)  # Dummy grid dataset approximated by GP as ground truth


#####################################################
# 03 REAL sparse and dense datasets read-in
#####################################################







#####################################################
# 04 Georeferencing REAL datasets
#####################################################






#####################################################
# 05 REAL Training, testing and validation
#####################################################




