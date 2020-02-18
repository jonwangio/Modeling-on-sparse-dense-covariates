#-------------------------------------------------------------------------------
# Name:        Learning spatial processes through sparse measurements 
#              and its dense gridded covariate(s)

##########
# Ground truth function
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
import matplotlib.pyplot as plt
import GPy


#####################################################
# 01 Create dummy sparse and dense datasets
#####################################################

# Grid dummy dataset with specified function over input space with rows (r) and columns (c).
# The grid dataset is then approximated by 2D (combined) Gaussian Process as a ground truth process.
# The ground truth process is then used to test and validate different models.

#==================================
# 01_1 Real function based dummy dataset
#==================================
# Function options for creating dummy dataset over input space

##############
# Possible functions
##############
# Branin function
def branin(X):
    y = (X[:,1]-5.1/(4*np.pi**2)*X[:,0]**2+5*X[:,0]/np.pi-6)**2
    y += 10*(1-1/(8*np.pi))*np.cos(X[:,0])+10
    return (y)

# Prior random Gaussian Processes
# Arbitrary kernel definition
def arbKer(a, b, var, gamma):
    # Squared exponential kernel
    sqdist = np.sum(a**2,1).reshape(-1,1) + np.sum(b**2,1) - 2*np.dot(a, b.T)
    return var*np.exp(-0.5*sqdist/(gamma**2))

def priorGP(X, var, gamma):
    # GP controlled noise drawn from a covariance matrix generated from kernel
    K = arbKer(X, X, var, gamma)
    # Cholesky decomposition
    L = np.linalg.cholesky(K + 1e-6*np.eye(K.shape[0]))
    # Draw sample (can be multiple)
    Y = np.dot(L, np.random.normal(size=(X.shape[0],1)))  # Draw 1 sample
    return (Y)
# ...

# Dummy grid dataset defined over input space
def grid(x1min, x1max, x2min, x2max, row, col, var, gamma):
    xg1 = np.linspace(x1min,x1max,row)
    xg2 = np.linspace(x2min,x2max,col)

    X = np.zeros((xg1.size * xg2.size,2))
    for i,x1 in enumerate(xg1):
        for j,x2 in enumerate(xg2):
            X[i+xg1.size*j,:] = [x1,x2]
    # From surface function options
    #Y = branin(X)
    Y = priorGP(X, var, gamma)
    return (X, Y)  #Y[:,None])

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
def kern():
    kg = GPy.kern.RBF(input_dim=2, ARD=True)
    #kb = GPy.kern.Bias(input_dim=2)
    k = kg #+ kb
    k.plot()
    return (k)

# Model specification
def model(X, Y):
    k = kern()  # Kernel
    
    m = GPy.models.GPRegression(X,Y,k,normalizer=True)  # Specify model
    #m.sum.bias.variance.constrain_bounded(1e-3,1e5)
    #m.sum.rbf.variance.constrain_bounded(1e-3,1e5)
    m.rbf.variance.constrain_bounded(1e-3,1e5)
    #m.sum.rbf.lengthscale.constrain_bounded(.1,200.)
    m.rbf.lengthscale.constrain_bounded(.1,200.)
    m.Gaussian_noise.variance.constrain_fixed(1e-3, 1e-1)
    
    m.randomize()  # Random initialization
    m.optimize()  # Optimization
    m.plot()
    return (m)

# GP realization of dummy function-based dataset as ground truth GP
def gtGP(X, Y, s):
    r, c = int(np.sqrt(X.shape[0])*s), int(np.sqrt(X.shape[0])*s)  # Rows and columns with scaled density
    m = model(X, Y)  # Train the model over dummy grid dataset
    print(m)
    
    xg1 = np.linspace(X[:,0].min(),X[:,0].max(),r)
    xg2 = np.linspace(X[:,1].min(),X[:,1].max(),c)

    Xp = np.zeros((xg1.size * xg2.size,2))
    for i,x1 in enumerate(xg1):
        for j,x2 in enumerate(xg2):
            Xp[i+xg1.size*j,:] = [x1,x2]
    # Draw dense GP approximation to the dummy through trained model
    Yp = m.predict(Xp)[0]
    return (Xp, Yp, m)
    
# Uncertainty/accuracy control of the GP realization: ground truth GP needs to be GOOD!
def accGP():
    return None


#==================================
# 02_1 Inference through Coregionalized Gaussian processes (Multi-task GP/CoKriging)
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
                  
    K = GPy.kern.RBF(2, ARD=True)  # 2-D Radial Basis Function
    icm = GPy.util.multioutput.ICM(input_dim=2,num_outputs=2,kernel=K)
    m = GPy.models.GPCoregionalizedRegression(X,Y,kernel=icm)    
    m.optimize()
    print(m)
    W = m.ICM.B.W
    B = W*W.T  # Covariance matrix
    Bnorm = (B/np.sqrt(np.diag(B))).T/np.sqrt(np.diag(B))  # Correlation as normalized covariance matrix
    print('The correlation matrix is \n', Bnorm)
    return (m, Bnorm)

