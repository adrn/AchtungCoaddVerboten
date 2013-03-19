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
    
    def __repr__(self):
        return "<DCStar ({:0.2f},{:0.2f}), Flux={:0.1f}, σ={:.2f}>".format(self.x0, self.y0, self.flux, self.sigma)
    
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
    
    def __add__(self, other):
        """ Add the star to the image. Addition only supported with 
            DCImage objects.
        """
        
        if not isinstance(other, DCImage):
            raise TypeError("Addition only supported with DCImage objects.")
            
        new_image = other.copy()
        
        new_image.star = self
        new_image.data += util.gaussian(flux=self.flux, 
                                              position=(self.x0,self.y0),
                                              sigma=self.sigma,
                                              shape=other.shape)
        
        logger.debug("Added {star} to {img}".format(img=other, star=self))
        
        return new_image
    
    def __radd__(self, other):
        return self.__add__(other)

class DCGaussianNoiseModel(object):
    
    def __init__(self, sigma, sky_level):
        """ A Gaussian noise model for images with given std. dev. sigma
            and a uniform sky brightness sky_level.
        """
        

class DCImage(object):
    """ A simulated astronomical image. """
    
    def __repr__(self):
        return "<DCImage {shape[0]}x{shape[1]} id={id}>".format(shape=self.shape, id=self.id)
    
    def __init__(self, shape, id=None):
        """ Parameters
            ----------
            shape : tuple
                Should be a tuple of (x,y) values representing the xsize, ysize of the image
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
    
    def __hash__(self):
        return hash((self.id,self.shape[0], self.shape[1]))
        
    def add_noise(self, noise_model):
        """ Add some noise model to the data.
        """
        
        # Add noise to image data
        self.sigma = np.sqrt(self.sigma**2 + sigma**2)
        self._add_gaussian_noise(sigma)
        
        # Add some sky brightness
        self.sky_level += sky_level
        self.data += sky_level
        
        logger.debug("Added noise to {} with sky={:0.2f}, σ={:.2f}".format(self, sky_level, sigma))
    
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
        copied = copy.deepcopy(self)
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
    
    def save(self, filename=None, min=None, max=None, clip=True):
        """ Show the image as a bitmap using PIL """
        
        if filename == None:
            filename = "index{}.png".format(self.index)
        
        return util.save_image_data(self.data, filename=filename, min=min, max=max, clip=clip)
    
    @property
    def star_model(self):
        k = self.shape[0] // 2 + 1
        star_data = util.gaussian_star(position=(k,k), flux=self.star.flux, sigma=self.star.sigma, shape=self.shape)
        return star_data
    
    def centroid_star(self, gridsize=3, plot=False):
        """ Given noisy image data, and the 'model' image data (noiseless image),
            compute the chi-squared at all positions on a 3x3 grid around the nominal
            (true, known) position of the star.
        """
        
        chisq = util.position_chisq(self, gridsize=gridsize)
        params = util.fit_surface(chisq)
        
        if plot:
            xx,yy = np.meshgrid(range(gridsize), range(gridsize))
            
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.plot_wireframe(xx+7, yy+7, chisq)
            ax.plot([self.star.x0,self.star.x0],[self.star.y0,self.star.y0], [chisq.min(), chisq.max()], 'r-')
            #fig.savefig("images/centroid_chisq_{}coadded.png".format(self.index+1))
        
        x0,y0 = util.surface_maximum(params)
        d = (self.shape[0] - gridsize) / 2 + 1
        
        """
        print self.shape
        print d
        print x0,y0
        try:
            print self.star.x0, self.star.y0
        except:
            print self.stars[0].x0, self.stars[0].y0
        sys.exit(0)
        """
        
        logger.debug("\t\tFound star at: {}".format((x0+d, y0+d)))
        return (x0+d, y0+d)
    
    def copy(self):
        return copy.copy(self)

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
    