# coding: utf-8
from __future__ import division

""" """

# Standard library
import os, sys
import copy
import logging

# Third-party
import Image
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter

# Project
import util

# ==================================================================================================

# Create logger
logger = logging.getLogger(__name__)
ch = logging.StreamHandler()
formatter = logging.Formatter("%(name)s / %(levelname)s / %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)

class DCStar(object):
    
    def __init__(self, position, flux, sigma):
        """ A simulated observation of a star.
        
            Parameters
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
        
        logger.debug("Created new DCStar : {}".format(self))
    
    def __repr__(self):
        return "<DCStar ({:0.2f},{:0.2f}), Flux={:0.1f}, σ={:.2f}>".format(self.x0,
                                                                           self.y0,
                                                                           self.flux,
                                                                           self.sigma)
    
    def image(self, shape):
        """ Return a 2D array with the star. """
        return util.gaussian(flux=self.flux, 
                             position=(self.x0,self.y0),
                             sigma=self.sigma,
                             shape=shape)
    
    def __add__(self, other):
        """ Add the star to the image. Addition only supported with 
            DCImage objects.
        """
        
        if not isinstance(other, DCImage):
            raise TypeError("Addition only supported with DCImage objects.")
            
        new_image = other.copy()
        
        new_image.star = self
        new_image.data += self.image(new_image.shape)
        
        logger.debug("Added {star} to {img}".format(img=other, star=self))
        
        return new_image
    
    def __radd__(self, other):
        return self.__add__(other)

class DCGaussianNoiseModel(object):
    
    def __init__(self, sigma, sky_level):
        """ A Gaussian noise model for images with given std. dev. sigma
            and a uniform sky brightness sky_level.
        """
        self.sigma = sigma
        self.sky_level = sky_level
    
    def image(self, shape):
        """ Return a 2D image with the given noise properties """
        return np.random.normal(self.sky_level, self.sigma, shape)
    
    def __add__(self, other):
        """ Add this noise model to the image. """
        
        if not isinstance(other, DCImage):
            raise TypeError("Addition only supported with DCImage objects.")
            
        new_image = other.copy()
        new_image.sigma = np.sqrt(self.sigma**2 + other.sigma**2)
        new_image.sky_level += self.sky_level
        new_image.data += self.image(new_image.shape)
        
        logger.debug("Added noise to {} with sky={:0.2f}, σ={:.2f}".format(self, 
                        self.sky_level, self.sigma))

    def __radd__(self, other):
        return self.__add__(other)

class DCImage(object):
    """ A simulated astronomical image. """
    
    def __init__(self, shape, id=None):
        """ Parameters
            ----------
            shape : tuple
                Should be a tuple of (x,y) values representing the xsize, 
                ysize of the image
            id : any hashable object
                Within a trial, the ID of the image
        """
        
        try:
            if len(shape) != 2:
                raise ValueError("'shape' must be a length 2 tuple.")
        except TypeError:
            raise TypeError("'shape' must be a length 2 tuple or iterable.")
        
        # Set defaults
        self.shape = shape
        self.id = id
        self.sky_level = 0.0
        self.sigma = 0.0
        
        # Create zeroed image array
        self.data = np.zeros(self.shape, dtype=float)
        
        logger.debug("Created new DCImage : {}".format(self))
    
    def __repr__(self):
        return "<DCImage {shape[0]}x{shape[1]} id={id}>".format(shape=self.shape, 
                                                                id=self.id)
    
    def __hash__(self):
        return hash((self.id,self.shape[0], self.shape[1]))
    
    @property
    def size(self):
        return self.shape[0]*self.shape[1]
    
    @property
    def SN2(self):
        return 1.0 / (self.star.sigma * self.sigma)**2
    
    def smoothed(self, sigma):
        """ Return a smoothed copy of the current image instance. """
        
        smoothing_sigma = np.sqrt(sigma**2 - self.star.sigma**2)
        
        # Return smoothed copy of the image
        copied = self.copy()
        copied.data = gaussian_filter(self.data-self.sky_level, smoothing_sigma)
        copied.star.sigma = sigma
        
        return copied
    
    def gremlinized(self, amplitude):
        """ Return a gremlinized copy of the current image instance. Gremlinized means
            our knowledge of the true read noise and star spread gets obscured by some
            factor ('amplitude'), and our model image (the noiseless star image) is
            changed to represent what we *think* the source is.
        """
            
        copied = copy.copy(self)
        copied.sigma = self.sigma * np.random.uniform(1-amplitude, 1+amplitude)
        copied.star.sigma = self.star.sigma * np.random.uniform(1-amplitude, 1+amplitude)
        
        return copied
    
    def show(self, ax=None):
        """ Show the image using matplotlib """
        if ax == None:
            plt.imshow(self.data, cmap=cm.gray, interpolation="none")
            plt.show()
        
        else:
            ax.imshow(self.data, cmap=cm.gray, interpolation="none")
            return ax
    
    def save(self, filename=None, **kwargs):
        """ Save the image as a bitmap using PIL """
        
        if filename == None:
            filename = "index{}.png".format(self.index)
        
        return util.save_image_data(self.data, filename=filename, **kwargs)
    
    def copy(self):
        return copy.copy(self)

class DCCoaddmaschine(object):
    
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
            self.weight_images = np.ones((len(images), ) + self.shape)
        else:
            self.weight_images = np.array([np.ones(self.shape)*getattr(image, weight_by) for image in images])
        
        # Sum images with weights
        image_data = np.array([image.data-image.sky_level for image in images])
        self.data = np.sum(self.weight_images*image_data, axis=0) / np.sum(self.weight_images, axis=0)
        
        # APW: HACK!
        #star_model_data = np.array([image.star_model_data for image in images])
        #self.star_model_data = np.sum(weight_images*star_model_data, axis=0) / np.sum(weight_images, axis=0)
        
        self.stars = [image.star for image in images]
    
    @property
    def star_model(self):
        k = self.shape[0] // 2 + 1
        star_data = np.array([util.gaussian_star(position=(k,k), flux=star.flux, sigma=star.sigma, shape=self.shape) for star in self.stars])
        return np.sum(self.weight_images*star_data, axis=0) / np.sum(self.weight_images, axis=0)
    
    @property
    def SN2(self): return None
    def add_star(self): return None
    def add_noise(self): return None
    def gremlinized(self): return None
        
def cumulative_coadd(images, weight_by=None):
    """ Cumulatively coadd the image data with an optional weight per image """
    
    coadded_images = []
    for ii in range(len(images)):
        coadded_images.append(DCCoaddedImage(images[:ii+1], index=ii, weight_by=weight_by))
    
    return coadded_images
    