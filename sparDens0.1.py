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
from random import random
from math import cos, sin, floor, sqrt, pi, ceil
from mpl_toolkits.mplot3d import Axes3D  


#####################################################
# 01 1D Toy Gaussian Process
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
def grid(x1min, x1max, x2min, x2max, yr, c):
    xg1 = np.linspace(x1min,x1max,r)
    xg2 = np.linspace(x2min,x2max,c)

    X = np.zeros((xg1.size * xg2.size,2))
    for i,x1 in enumerate(xg1):
        for j,x2 in enumerate(xg2):
            X[i+xg1.size*j,:] = [x1,x2]
    Y = branin(X)  # Call surface function
    return(X, Y[:,None])

# Show dummy grid dataset
def showGrid(X, Y):
    r, c = int(np.sqrt(X.shape[0])), int(np.sqrt(X.shape[0]))  # Rows and columns
    x1, x2 = X.reshape(r,c,2)[:,:,0], X.reshape(r,c,2)[:,:,1]
    Y = Y.reshape(r,c)
    print('Surface dimensions in x1, x2, and y are ', x1.shape, x2.shape, Y.shape)
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
def gtGP(X, Y, s):
    r, c = int(np.sqrt(X.shape[0])*s), int(np.sqrt(X.shape[0])*s)  # Rows and columns with scaled density
    m = mGP(X, Y)  # Train the model over dummy grid dataset
    print(m)
    
    xg1 = np.linspace(-5,10,r)
    xg2 = np.linspace(0,15,c)

    Xp = np.zeros((xg1.size * xg2.size,2))
    for i,x1 in enumerate(xg1):
        for j,x2 in enumerate(xg2):
            Xp[i+xg1.size*j,:] = [x1,x2]
    # Draw dense GP approximation to the dummy through trained model
    Yp = m.predict(Xp)[0]
    return(Xp, Yp, m)
    
# Uncertainty/accuracy control of the GP realization: ground truth GP needs to be GOOD!
def accGP():
    return None

    
#==================================
# 02_3 Sparse point observation from the ground truth processes
#==================================
# TWO OPTIONS!!!
# 1_perturbate the ground truth GP before sampling
# 2_perturbate the randomly sampled points from the GP
# In either way, we need two types of functions: 
# (1) random points generation functions, and (2) perturbation functions
    
# Plot point in 3d
def plotPt(x, y):
    # Plot 3d random points
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.set_xlim3d(np.min(x[:,0]),np.max(x[:,0]))
    ax.set_ylim3d(np.min(x[:,1]),np.max(x[:,1]))
    ax.set_zlim3d(np.min(y),np.max(y))
    ax.scatter(x[:,0], x[:,1], y)
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('Y')
    plt.show()
    return None
    
# Random point sample from the grid surface
def randPt(X, Y, n):
    dim = X.shape[0]  # Input data dimension
    randInd = np.random.randint(0,dim,size=n)  # Index of n random draw
    
    x = X[randInd,:]  # Random draw of X
    y = Y[randInd,:]  # Random draw of Y
    
    # Plot 3d random points
    plotPt(x, y)
    return (x, y)


# Poisson-disc distribution random sampling
# r as minimal distance between points, k as number of points, dist as euclidian distance function
def poissonPt(X, Y, r, k, random=random):
    width, height = int(np.sqrt(X.shape[0])), int(np.sqrt(X.shape[0]))  # Rows and columns as number grids
    tau = 2 * pi
    cellsize = r / sqrt(2)  # Cellsize defined to draw one point

    grid_width = int(ceil(width / cellsize))  # Number of grid as width
    grid_height = int(ceil(height / cellsize))  # Number of grid as height
    grid = [None] * (grid_width * grid_height)
    
    def dist(a, b):  # Euclidean distance
        dx = a[0] - b[0]
        dy = a[1] - b[1]
        return sqrt(dx * dx + dy * dy)
    
    def grid_coords(p):  # Grid location as grid index
        return int(floor(p[0] / cellsize)), int(floor(p[1] / cellsize))

    def fits(p, gx, gy):
        yrange = list(range(max(gy - 2, 0), min(gy + 3, grid_height)))
        for x in range(max(gx - 2, 0), min(gx + 3, grid_width)):
            for y in yrange:
                g = grid[x + y * grid_width]
                if g is None:
                    continue
                if dist(p, g) <= r:
                    return False
        return True

    p = width * random(), height * random()
    queue = [p]
    grid_x, grid_y = grid_coords(p)
    grid[grid_x + grid_y * grid_width] = p

    while queue:
        qi = int(random() * len(queue))
        qx, qy = queue[qi]
        queue[qi] = queue[-1]
        queue.pop()
        for _ in range(k):
            alpha = tau * random()
            d = r * sqrt(3 * random() + 1)
            px = qx + d * cos(alpha)
            py = qy + d * sin(alpha)
            if not (0 <= px < width and 0 <= py < height):
                continue
            p = (px, py)
            grid_x, grid_y = grid_coords(p)
            if not fits(p, grid_x, grid_y):
                continue
            queue.append(p)
            grid[grid_x + grid_y * grid_width] = p
    
    # Project grid index p to index of column shaped X
    p = [p for p in grid if p is not None]
    p = np.array(p).astype(int)
    ind = (p[:,0]-1) * width + p[:,1]
    
    x = X[ind,:]  # Random draw of X
    y = Y[ind,:]  # Random draw of Y
    
    # Plot 3d random points
    plotPt(x, y)
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
    plotPt(x, y)
    return (x, y)

        
 
#==================================
# 02_4 Dense covariate from the ground truth processes
#==================================
# One handy option is to generate covariate(s) through linear transformation.
# Other options can be insufficient observations or other components to be involved.
# The most ideal result should give a strong free form coregionalization matrix
# B = WW', where, as B is normalized to correlation, the diagonal is close to 1.

# Generate dense covariate through linear transformation of the ground truth GP
# Scale, shift in x1, x2, and y, convolution and etc..
def linCov(X,Y):
    scale = np.linspace(-2,2,1)
    for s in scale:
        print('Scale Y by ', s)
        Ys = Y*s
        showGrid(X, Ys)
    return (X, Ys)


# Generate dense covariate as a insufficient observation of the ground truth GP
def insuffCov(X,Y,n):
    # Insufficiently observed points
    x, y = randPt(X,Y,n)
    # A proximation of the ground truth through few points
    Xp, Yp, m = gtGP(x, y, r, c)  # Take advantage of the gtGP function
    

# Generate dense covariate as noisy versions of the ground truth GP
def noiseCov(X, Y, m, s):
    # Simple noise follows N(mean, std**2)
    mean = m
    std = s
    Ynoise = std*np.random.randn(Y.size, 1)+mean
    Y += Ynoise  # Call surface function
    showGrid(X, Y)
    return(X, Y)
    
# More advanced noise from the "Colors of Noise"
def noiseBasis():
    
    return None

def noiseColorCov():
    
    return None
    

# Generate dense covariate with advanced GP controlled noise
# Kernel definition
def ker(a, b, var, lenscale):
    # Squared exponential kernel
    var = .5
    sqdist = np.sum(a**2,1).reshape(-1,1) + np.sum(b**2,1) - 2*np.dot(a, b.T)
    return np.exp(-var*sqdist)

def noiseGPCov(X, Y, var, lenscale):
    # GP controlled noise drawn from a covariance matrix generated from kernel
    K = ker(X, X)
    # Cholesky decomposition
    L = np.linalg.cholesky(K + 1e-6*np.eye(K.shape[0]))
    # Draw sample (can be multiple)
    Y_prior = np.dot(L, np.random.normal(size=(X.shape[0],1)))  # Draw 1 sample
    Y += Y_prior
    showGrid(X, Y)
    showGrid(X, Y_prior)
    return(X, Y_prior)

    
#==================================
# 02_5 Coregionalized processes (Multi-task GP/CoKriging) investigation
#==================================

# Learn coregionalization model
def coregionGP(X0, Y0, X1, Y1):
#    X0widx = np.c_[X0,np.ones(X0.shape[0])*0]  # Add a column of coregionalized index through np.c_
#    X1widx = np.c_[X1,np.ones(X1.shape[0])*1]
#    X = np.r_[X0widx,X1widx]  # Row-wise merge all X through np.r_
#    Y = np.r_[Y0,Y1]
#    kern = GPy.kern.RBF(input_dim=1)**GPy.kern.Coregionalize(input_dim=2,output_dim=2, rank=1)
#    m = GPy.models.GPCoregionalizedRegression(X,Y,kern)

    X = np.array([X0, X1])  # Prepare X and Y for Intrinsic Coregionalization Model (ICM)
    Y = np.array([Y0, Y1])
                  
    K = GPy.kern.RBF(2)  # 2-D Radial Basis Function
    icm = GPy.util.multioutput.ICM(input_dim=2,num_outputs=2,kernel=K)
    m = GPy.models.GPCoregionalizedRegression(X,Y,kernel=icm)    
    m.optimize()
    print(m)
    W = m.ICM.B.W
    B = W*W.T
    print('The correlation matrix is')
    (B/np.sqrt(np.diag(B))).T/np.sqrt(np.diag(B))
    return (m)



#==================================
# 02_5 Section '__main__'
#==================================
# Dummy gridded dataset over input space
x1min, x1max, x2min, x2max = -5, 10, 0, 15  # Domain of the input space
r, c = 30, 30  # Grid number of the input space 
X, Y = grid(x1min,x1max,x2min,x2max,r,c)  # Dummy grid dataset realized by function
showGrid(X, Y)  # Show function dummy grid

# Ground truth representation through GP (parameters are var and lengthscale)
scale = 2   # Define GP prediction/approximation space as densified input space
Xp, Yp, m = gtGP(X, Y, scale)  # Approximation
showGrid(Xp, Yp)  # Dummy grid dataset approximated by GP as ground truth

# Point samples as point observation
k = 200  # Number of point observations
r = 2  # Minimal distance (number of grid) between points
x, y = poissonPt(X, Y, r, k, random=random)

# Dense covariate(s) with noise
Xcov, Ycov = linCov(Xp, Yp)  # Dense covariates through linear transformation

# Prediction/modeling test through GP Coregionalization
m_co = coregionGP(x, y, Xcov, Ycov)  # Coregionalization model


#####################################################
# 03 REAL sparse and dense datasets read-in
#####################################################







#####################################################
# 04 Georeferencing REAL datasets
#####################################################






#####################################################
# 05 REAL Training, testing and validation
#####################################################




