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
    
    def __init__(self, position, flux, sigma, shape):
        """ A simulated observation of a star.
        
            Parameters
            ----------
            position : tuple
                The pixel coordinates of the star.
            flux : float
                Some measure of the brightness of the star in the image
            sigma : float
                Some measure of the spread of the star -- seeing + PSF
        """
        
        self.x0, self.y0 = position
        self.flux = flux
        self.sigma = sigma
        
        self.data = util.gaussian(flux=self.flux, 
                                  position=(self.x0,self.y0),
                                  sigma=self.sigma,
                                  shape=shape)
        
        logger.debug("Created new DCStar : {}".format(self))
    
    def __repr__(self):
        return "<DCStar flux={:0.1f}, σ={:.2f}>".format(self.flux,
                                                        self.sigma)
    
    def __add__(self, other):
        """ Add the star to the image. Addition only supported with 
            DCImage objects.
        """
        
        if not isinstance(other, DCImage):
            raise TypeError("Addition only supported with DCImage objects.")
            
        new_image = other.copy()
        new_image.star = self
        new_image.data += self.data
        logger.debug("Added {star} to {img}".format(img=other, star=self))
        
        return new_image
    
    def __radd__(self, other):
        return self.__add__(other)

class DCGaussianNoiseModel(object):
    
    def __init__(self, sigma, sky_level, shape):
        """ A Gaussian noise model for images with given std. dev. sigma
            and a uniform sky brightness sky_level.
        """
        self.sigma = sigma
        self.sky_level = sky_level
        self.data = np.random.normal(self.sky_level, self.sigma, shape)
    
    def __add__(self, other):
        """ Add this noise model to the image. """
        
        if not isinstance(other, DCImage):
            raise TypeError("Addition only supported with DCImage objects.")
            
        new_image = other.copy()
        new_image.sigma = np.sqrt(self.sigma**2 + other.sigma**2)
        new_image.sky_level += self.sky_level
        new_image.data += self.data
        logger.debug("Added noise to {} with sky={:0.2f}, σ={:.2f}".format(self, 
                        self.sky_level, self.sigma))
        
        return new_image

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
            our knowledge of the true read noise and star spread is wrong by some
            amount (specified with 'amplitude'). Our model image (the noiseless 
            star image) is changed to represent what we *think* the source is.
        """
            
        copied = copy.copy(self)
        copied.sigma = self.sigma * np.random.uniform(1.-amplitude, 1.+amplitude)
        copied.star.sigma = self.star.sigma * np.random.uniform(1.-amplitude, 1.+amplitude)
        
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
    
    def __init__(self, images):
        """ Das Coaddmaschine is created with a list of DCImage objects, and 
            will return cumulative coadds of the images with specified weights
            and ordering.
        """
        self.images = images
    
    def _sort_indices(self, sort_by=None):
        """ """
        if sort_by == "sn2":
            indices = np.argsort([image.SN2 for image in self.images])[::-1]
        elif sort_by == "psf":
            indices = np.argsort([image.star.sigma for image in self.images])
        elif sort_by == None:
            indices = range(len(self.images))
        else:
            raise ValueError("sort_by can only be 'psf' or 'sn2'")
        
        return indices
    
    def sorted_images(self, sort_by=None):
        """ Return the images sorted by some """
        indices = self._sort_indices(sort_by=sort_by)
        return [self.images[idx] for idx in indices]
    
    def weight_images(self, sorted_images, weight_by=None):
        """ """
        if weight_by == None:
            weights = np.ones(len(self.images))
        elif weight_by == "sn2":
            weights = np.array([image.SN2 for image in sorted_images])
        
        weight_images = []
        for ii in range(len(sorted_images)):
            weight_images.append(np.ones(self.images[0].shape)*weights[ii])
        
        return np.array(weight_images)
    
    def coadd(self, weight_by=None, sort_by=None):
        """ Return a cumulative coadd ... """

        sorted_images = self.sorted_images(sort_by=sort_by)
        weight_images = self.weight_images(sorted_images, weight_by=weight_by)
        
        coadded_images = []
        for ii in range(len(sorted_images)):
            image_subset = sorted_images[:ii+1]
            
            individual_data = np.array([(image.data-image.sky_level) 
                                         for image in image_subset])*weight_images[:ii+1]
            coadded_data = np.sum(individual_data, axis=0) / np.sum(weight_images[:ii+1], axis=0)
            
            # Compute the new variance as a sum in quadrature 
            sigma = np.sqrt(np.sum([image.sigma for image in image_subset]))
            
            coadded_image = DCImage(self.images[0].shape, id=ii+1)
            coadded_image.sigma = sigma
            coadded_image.data = coadded_data
            coadded_image.sky_level = 0.0
            coadded_image.star = image.star
            coadded_images.append(coadded_image)
        
        return coadded_images
        
    def coadd_models(self, sort_by=None, weight_by=None):
        sorted_images = self.sorted_images(sort_by)
        weight_images = self.weight_images(sorted_images, weight_by=weight_by)
        sorted_star_data = [image.star.data for image in sorted_images]
        
        coadded_models = []
        for ii in range(len(sorted_star_data)):
            c = np.sum(weight_images[:ii+1]*sorted_star_data[:ii+1], axis=0) / np.sum(weight_images[:ii+1], axis=0)
            
            im = DCImage(sorted_images[0].shape, id=ii+1)
            im.sigma = 0.
            im.sky_level = 0.
            im.data = c
            coadded_models.append(im)
            
        return coadded_models
    
