# coding: utf-8
from __future__ import division

""" Utility functions and misc. """

# Standard library
import os, sys
import logging

# Third-party
import Image
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import scipy.optimize as so

def gaussian(position, flux, sigma, shape):
    """ Return an image of a Gaussian star.
        
        Parameters
        ----------
        position : tuple
            The center of the star (x0,y0).
        flux : float
            The flux value or normalization.
        sigma : float
            The spread of the star.
        shape : tuple
            The size of the meshgrid to put the star on.
    """
    x0,y0 = position
    x = np.arange(shape[0])
    y = np.arange(shape[1])
    x_grid, y_grid = np.meshgrid(x,y)
    return flux / (2.*np.pi*sigma**2) * np.exp(- ((x_grid-x0)**2 + (y_grid-y0)**2) / (2*sigma**2))

def linear_scale(data, min=None, max=None, clip=True):
    """ Scale an array of data to values between 0 and 1
    
        Parameters
        ----------
        data : numpy.ndarray
            The numpy array to be rescaled.
        min : float (optional)
            The minimum pixel value -- gets mapped to 0.
        max : float (optional)
            The maximum pixel value -- gets mapped to 1.
        clip : bool (optional)
            If true, will clip values > 1 to 1 and < 0 to 0.
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

def save_image_data(data, filename, min=None, max=None, clip=True):
    """ Save a 2D array of image data as a png.
    
        Parameters
        ----------
        data : numpy.ndarray
            A 2D numpy array of image data.
        filename : str
            A full filename (path + filename) to save the bitmap image to.
        min : float (optional)
            The minimum pixel value -- gets mapped to 0.
        max : float (optional)
            The maximum pixel value -- gets mapped to 1.
        clip : bool (optional)
            If true, will clip values > 1 to 1 and < 0 to 0.
    """
    scaled = 255.*linear_scale(data, min=min, max=max, clip=clip)
    
    im = Image.fromarray(scaled.astype(np.uint8))
    im.save(filename)
    
    return min, max

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

def position_chisq(image, gridsize, test=False):
    """ Compute the chisquared value for placing the model in every position
        on a 3x3 grid around the aligned positions.
    """
    
    f = gridsize-1
    g = np.ceil(gridsize / 2)
    
    chisq = np.zeros((gridsize,gridsize), dtype=float)
    data_cutout = image.image_data[g:-g,g:-g] - image.sky_level
    
    star_model = image.star_model
    shp = star_model.shape
    
    for ii in range(gridsize):
        for jj in range(gridsize):
            star_cutout = star_model[ii:shp[0]+(ii-f),jj:shp[1]+(jj-f)]
            chisq[ii,jj] = np.sum((data_cutout-star_cutout)**2) / image.sigma**2

    return np.flipud(np.fliplr(chisq))

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

def quad_surface_model(params, x, y):
    a, b, c, d, e, f = params
    return a + (b * x) + (c * y) + (d * x ** 2) + (2.0 * e * x * y) + (f * y ** 2)

def quad_surface_error_function(params, x, y, data):
    return surface_model(params, x, y) - data

def fit_quad_surface(data):
    """ Use scipy.optimize.leastsq to fit a 2D quadratic surface 
        to 3D data.
    
        Parameters
        ----------
        data : numpy.ndarray
            A 2D array of values.
            
    """
    
    r = data.shape[0]
    xx, yy = np.meshgrid(range(r), range(r))
    xx = xx.astype(float)
    yy = yy.astype(float)
    
    fit_parameters, ier = so.leastsq(error_function, \
                                    [0.]*6, \
                                    args=(xx.ravel(), yy.ravel(), data.ravel()), \
                                    maxfev=10000, \
                                    full_output=False)

    return fit_parameters

def quad_surface_maximum(params):
    """ Analytically compute the maximum of a 2D surface, given the 
        parameters of the quadratic model.
        
        Parameters
        ----------
        params : iterable
            'Best' model parameters for a quadratic surface.
    
    """
    a,b,c,d,e,f = params
    x = (c*e - b*f) / (2.0*d*f - 2.0*e**2)
    y = (b*e - c*d) / (2.0*d*f - 2.0*e**2)
    return x,y