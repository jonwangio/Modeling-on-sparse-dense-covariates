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

from mpl_toolkits.mplot3d import Axes3D  

#########################
# Toy Gaussian Process in 1-dimensional
#########################

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


#########################
# Create dummy sparse and dense datasets
#########################

# Grid dummy dataset with specified function over input space with rows (r) and columns (c).
# The grid dataset is then approximated by 2D (combined) Gaussian Process as a ground truth process.
# The ground truth process is then used to test and validate different models.

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
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    print(x1.shape, x2.shape, Y.shape)
    ax.plot_surface(x1, x2, Y)
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('Y')
    plt.show()
    return None

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
    
# Uncertainty/accuracy ensure of the GP realization: ground truth GP needs to be GOOD!
def accGP():
    

# TWO OPTIONS!!!
# 1_perturbate the ground truth GP before sampling
# 2_perturbate the randomly sampled points from the GP
# In either way, we need two types of functions: 
# (1) random points generation functions, and (2) perturbation functions
    

# Random point sample from the grid surface
def randPt(n):
    
    
original_data = np.random.rand(100,100)

fig, (ax, ax2) = plt.subplots(ncols=2)
im = ax.imshow(original_data, cmap="summer")


N = 89
x = np.random.randint(0,100,size=N)
y = np.random.randint(0,100,size=N)

random_sample = original_data[x,y]
sc = ax2.scatter(x,y,c=random_sample, cmap=im.cmap, norm=im.norm)

ax2.set_aspect("equal")
ax2.set(xlim=ax.get_xlim(), ylim=ax.get_ylim())

fig.colorbar(sc, ax=[ax,ax2], orientation="horizontal")
plt.show()
    
# 
    

# Dummy gridded dataset over input space
r, c = 5, 5  # Define input space 
X, Y = grid(r,c)  # Dummy grid dataset realized by function
showGrid(X, Y, r, c)  # Show function dummy grid

rGP, cGP = 100, 100   # Define GP prediction/approximation space over input space
Xp, Yp = gtGP(X, Y, rGP, cGP)  # Approximation
showGrid(Xp, Yp, rGP, cGP)  # Dummy grid dataset approximated by GP as ground truth


#########################
# REAL sparse and dense datasets read-in
#########################







#########################
# Georeferencing REAL datasets
#########################






#########################
# REAL Training, testing and validation
#########################




