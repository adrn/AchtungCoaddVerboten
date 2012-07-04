""" """

# Standard library
import os, sys
import copy

# Third-party
import Image
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter

# Project
import util

class DCStar(object):
    """ A simulated observation of a star """
    
    def __init__(self, position, flux, sigma):
        """ Parameters
            ----------
            position : tuple
                Should be a tuple of (x,y) values
            flux : float
                Some measure of the brightness of the star in the image
            sigma : float
                Some measure of the spread of the star -- seeing + PSF
        """
        
        self.x0, self.y0 = position
        self.flux = flux
        self.sigma = sigma

class DCImage(object):
    """ A simulated image with a single point-source (star) """
    
    def __init__(self, shape, index=None):
        """ Parameters
            ----------
            shape : tuple
                Should be a tuple of (x,y) values representing the xsize, ysize of the image
            index : int
                Within a trial, the index of the image
        """
                
        # Set default values for missing parameters
        if index == None:
            index = 0
        
        self.shape = shape
        self.index = index
        self.sky_level = 0.0
        
        # Create zeroed image array
        self.image_data = np.zeros(self.shape, dtype=float)
        
    def add_noise(self, sigma, sky_level):
        """ Add Gaussian noise to the image with the given root-variance 'sigma' 
            and a uniform sky brightness 'sky_level' 
        """
        
        # Add noise to image data
        self.sigma = sigma
        self.image_data += np.random.normal(0.0, sigma, self.shape)
        
        self.sky_level = sky_level
        self.image_data += sky_level
    
    def add_star(self, star):
        """ Add a DCStar to the image """
        self.star = star
        
        x = np.arange(self.shape[0])
        y = np.arange(self.shape[1])
        x_grid, y_grid = np.meshgrid(x,y)
        self.star_image_data = self.star.flux / (2.*np.pi*self.star.sigma**2) * np.exp(- ((x_grid-self.star.x0)**2 + (y_grid-self.star.y0)**2) / (2*self.star.sigma**2))
        self.image_data += self.star_image_data
    
    @property
    def size(self):
        return self.shape[0]*self.shape[1]
    
    @property
    def SN2(self):
        return 1.0 / (self.star.sigma**2 * self.sigma**2)
    
    def smoothed(self, sigma):
        """ Smooth the image by the specified ??? APW """
        
        smoothing_sigma = np.sqrt(sigma**2 - self.star.sigma**2)
        
        # Return smoothed copy of the image
        copied = copy.copy(self)
        copied.image_data = gaussian_filter(self.image_data-self.sky_level, smoothing_sigma)
        return copied
    
    def gremlinized(self, amplitude):
        copied = copy.copy(self)
        copied.sigma = self.sigma * np.random.uniform(1-amplitude, 1+amplitude)
        copied.star.sigma = self.star.sigma * np.random.uniform(1-amplitude, 1+amplitude)
        return copied
    
    def show(self, ax=None):
        """ Show the image using matplotlib """
        if ax == None:
            plt.imshow(self.image_data, cmap=cm.gray, interpolation="none")
            plt.show()
        
        else:
            ax.imshow(self.image_data, cmap=cm.gray, interpolation="none")
            return ax
    
    def save(self, filename=None, min=None, max=None, clip=True):
        """ Show the image as a bitmap using PIL """
        
        if filename == None:
            filename = "index{}.png".format(self.index)
        
        return util.save_image_data(self.image_data, filename=filename, min=min, max=max, clip=clip)

class DCCoaddedImage(DCImage):
    
    def __init__(self, images, weight_by=None, index=None):
        """ """
        
        # Set default values for missing parameters
        if index == None:
            index = 0
        
        self.index = index
        self.shape = images[0].shape
        self.sky_level = 0.0
        self.sigma = np.sqrt(np.sum([image.sigma for image in images]))
        
        # Figure out if the user wants to weight the images
        if weight_by == None:
            weight_images = np.ones((len(images), ) + self.shape)
        else:
            weight_images = np.array([np.ones(self.shape)*getattr(image, weight_by) for image in images])
        
        # Sum images with weights
        image_data = np.array([image.image_data-image.sky_level for image in images])
        self.image_data = np.sum(weight_images*image_data, axis=0) / np.sum(weight_images, axis=0)
        
        star_image_data = np.array([image.star_image_data for image in images])
        self.star_image_data = np.sum(weight_images*star_image_data, axis=0) / np.sum(weight_images, axis=0)
    
    @property
    def SN2(self): return None
    def add_star(self): return None
    def add_noise(self): return None
    def gremlinized(self): return None
        
def cumulative_coadd(images, weight_by=None):
    """ Cumulatively coadd the image data with an optional weight per image """
    
    coadded_images = []
    for ii in range(len(images)):
        coadded_images.append(DCCoaddedImage(images[:ii+1]))
    
    return coadded_images
    