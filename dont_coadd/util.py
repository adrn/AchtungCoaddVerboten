""" """

# Standard library
import os, sys
import logging

# Third-party
import Image
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import scipy.optimize as so

def save_image_data(data, filename, min=None, max=None, clip=True):
    """ Save an array of image data as a png 
    
        Parameters
        ----------
        data : numpy.ndarray
            A 2D numpy array of image data.
        filename : str
            A full filename (path + filename) to save the bitmap image to.
        min, max, clip (optional)
            Passed through to linear_scale().
    """
    scaled = 255.*linear_scale(data, min=min, max=max, clip=clip)
    
    im = Image.fromarray(scaled.astype(np.uint8))
    im.save(filename)
    
    return min, max

def linear_scale(data, min=None, max=None, clip=True):
    """ Scale an array of data to values between 0 and 1
    
        Parameters
        ----------
        data : numpy.ndarray
            An ND numpy array to be rescaled.
        min : float
            TODO
        max : float
            TODO
        clip : bool
            TODO
    """
    
    if min == None:
        min = data.min()
    
    if max == None:
        max = data.max()
    
    scaled = (data - min) / (max - min)
    
    if clip:
        scaled[scaled > 1] = 1.0
        scaled[scaled < 0] = 0.0
    
    return scaled

def plot_grid(images, filename):
    """ Takes an array of images and plots them on a grid, then saves to a file. """
    
    # Scale each image using the minimum and maximum values from the *first* image
    min = images[0].min()
    max = images[0].max()
    #min = images[-1].min()
    #max = images[-1].max()
    
    nrows = ncols = int(np.sqrt(len(images)))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12,12))
    
    for row in range(nrows):
        for col in range(ncols):
            index = row*ncols + col
            
            image_data = images[index]
                
            scaled_data = linear_scale(image_data, min=min, max=max)
            axes[row,col].imshow(scaled_data, cmap=cm.gray, interpolation="none")
            #axes[row,col].imshow(image_data, cmap=cm.gray, vmin=min, vmax=max)
            
            axes[row,col].get_xaxis().set_visible(False)
            axes[row,col].get_yaxis().set_visible(False)
            
            # Remove border?
            #axes[row,col].set_frame_on(False)
    
    plt.setp([a.get_xticklabels() for a in fig.axes], visible=False)
    plt.setp([a.get_yticklabels() for a in fig.axes], visible=False)
    plt.subplots_adjust(wspace=-0.4, hspace=0.0)
    fig.savefig(filename)

def position_chisq(image, gridsize):
    """ Compute the chisquared value for placing the model in every position
        on a 3x3 grid around the aligned positions.
    """
    
    f = gridsize-1
    g = np.ceil(gridsize / 2)
    
    chisq = np.zeros((gridsize,gridsize), dtype=float)
    data_cutout = image.image_data[g:-g,g:-g]
    shp = image.star_model_data.shape
    
    for ii in range(gridsize):
        for jj in range(gridsize):
            star_cutout = image.star_model_data[ii:shp[0]+ii-f,jj:shp[1]+jj-f] 
            chisq[ii,jj] = np.sum((data_cutout-star_cutout)**2) / image.sigma**2
    
    return chisq

'''
def gaussian2D(p, x, y):
    """ Returns a 2D Gaussian on the meshgrid x,y given the parameter array 'p'

    Parameters
    ----------
    p : numpy.ndarray, list, tuple
        p is a list of parameters for the Gaussian fit. 
          p[0] <-> flux
          p[1] <-> star sigma (rotationally symmetric)
          p[2] <-> x0, x coordinate of the center of the Gaussian
          p[3] <-> y0, y coordinate of the center of the Gaussian
          p[4] <-> sky, the background level of the Gaussian
    x : numpy.ndarray
        x is the result of a numpy meshgrid -- x,y = np.meshgrid(range(imageXSize),range(imageYSize))
    y : numpy.ndarray
        y is the result of a numpy meshgrid -- x,y = np.meshgrid(range(imageXSize),range(imageYSize))

    """
    return p[0] / (2.*np.pi*p[1]**2) * np.exp(-((x-p[2])**2 + (y-p[3])**2) / (2*p[1]**2)) + p[4]

def errorFunction(p, x, y, data):
    """ Returns the 'distance' of the model (with the given parameters 'p')
        to the data

    Parameters
    ----------
    p : numpy.ndarray, list, tuple
        p is a list of parameters for the Gaussian fit. 
          p[0] <-> flux
          p[1] <-> star sigma (rotationally symmetric)
          p[2] <-> x0, x coordinate of the center of the Gaussian
          p[3] <-> y0, y coordinate of the center of the Gaussian
          p[4] <-> sky, the background level of the Gaussian
    x : numpy.ndarray
        x is the result of a numpy meshgrid -- x,y = np.meshgrid(range(imageXSize),range(imageYSize))
    y : numpy.ndarray
        y is the result of a numpy meshgrid -- x,y = np.meshgrid(range(imageXSize),range(imageYSize))
    data : numpy.ndarray
        The image data to be compared to the model.
        
    """
    return gaussian2D(p, x, y) - data
    
def fit_2d_gaussian(data):
    r = data.shape[0]
    xx,yy = np.meshgrid(range(r), range(r))
    initialParameterGuess = [2., 1., r/2., r/2., 0.]
    
    fitParameters, ier = so.leastsq(errorFunction, \
                                    initialParameterGuess, \
                                    args=(xx.ravel(), yy.ravel(), data.ravel()), \
                                    maxfev=10000, \
                                    full_output=False)
    
    return fitParameters
'''

def surface_model(params, x, y):
    a, b, c, d, e, f = params
    return a + (b * x) + (c * y) + (d * x ** 2) + (2.0 * e * x * y) + (f * y ** 2)

def error_function(params, x, y, data):
    return surface_model(params, x, y) - data

def fit_surface(data):
    """ Fit a 2D surface to 3D data """
    
    r = data.shape[0]
    xx, yy = np.meshgrid(range(r), range(r))
    initialParameterGuess = [0.]*6
    
    fitParameters, ier = so.leastsq(error_function, \
                                    initialParameterGuess, \
                                    args=(xx.ravel(), yy.ravel(), data.ravel()), \
                                    maxfev=10000, \
                                    full_output=False)
    
    return fitParameters

def surface_maximum(params):
    """ Compute the maximum of a 2D surface, given the parameters of the model """
    a,b,c,d,e,f = params
    x = (c*e - b*f) / (2.0*d*f - 2.0*e**2)
    y = (b*e - c*d) / (2.0*d*f - 2.0*e**2)
    return x,y