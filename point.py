#-------------------------------------------------------------------------------
# Name:        Learning spatial processes through sparse measurements 
#              and its dense gridded covariate(s)

##########
# Points as point observations
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
from mpl_toolkits.mplot3d import Axes3D  
from random import random
from math import cos, sin, floor, sqrt, pi, ceil


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
# r as minimal distance separating points
# k as number of candidate points, 
# dist as euclidian distance function
def poissonPt(X, Y, r, k=10, random=random):
    r = r*np.sqrt(X.shape[0])/np.ptp(X[:,0])  # Real number distance converted to number of pixels
    width, height = int(np.sqrt(X.shape[0])), int(np.sqrt(X.shape[0]))  # Rows and columns as number of pixels
    tau = 2 * pi
    cellsize = r / sqrt(2)  # Cellsize defined over pixels to draw one point
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
    ind = (p[:,0]-1) * width + p[:,1]  # row, col index converted to column index in X and Y
    x = X[ind,:]  
    y = Y[ind,:]  
    
    # Plot 3d random points
    plotPt(x, y)
    return (x, y)

# More strategic sampling from characteristic locations (critical points?)
def cPt(X,Y,n):
    r, c = int(np.sqrt(X.shape[0])), int(np.sqrt(X.shape[0]))  # Rows and columns with scaled density
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

        
