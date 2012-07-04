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
from mpl_toolkits.mplot3d import axes3d
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
        self.star_model_data = self.star.flux / (2.*np.pi*self.star.sigma**2) * np.exp(- ((x_grid-self.star.x0)**2 + (y_grid-self.star.y0)**2) / (2*self.star.sigma**2))
        self.image_data += self.star_model_data
    
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
        copied.star_model_data = gaussian_filter(self.star_model_data, smoothing_sigma)
        
        return copied
    
    def gremlinized(self, amplitude):
        copied = copy.copy(self)
        copied.sigma = self.sigma * np.random.uniform(1-amplitude, 1+amplitude)
        copied.star.sigma = self.star.sigma * np.random.uniform(1-amplitude, 1+amplitude)
        
        # Replace star model image with gremlinized model
        x = np.arange(copied.shape[0])
        y = np.arange(copied.shape[1])
        x_grid, y_grid = np.meshgrid(x,y)
        copied.star_model_data = copied.star.flux / (2.*np.pi*copied.star.sigma**2) * np.exp(- ((x_grid-copied.star.x0)**2 + (y_grid-copied.star.y0)**2) / (2*copied.star.sigma**2))
        
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
    
    def centroid_star(self, gridsize=3, plot=False, ax=None):
        """ Given noisy image data, and the 'model' image data (noiseless image),
            compute the chi-squared at all positions on a 3x3 grid around the nominal
            (true, known) position of the star.
        """
        f = gridsize-1
        g = np.ceil(gridsize / 2)
        
        chisq = np.zeros((gridsize,gridsize), dtype=float)
        data_cutout = self.image_data[g:-g,g:-g]
        shp = self.star_model_data.shape
        
        for ii in range(gridsize):
            for jj in range(gridsize):
                star_cutout = self.star_model_data[ii:shp[0]+ii-f,jj:shp[1]+jj-f] 
                chisq[ii,jj] = np.sum((data_cutout-star_cutout)**2) / self.sigma**2
    
        #logging.debug("Smallest chi-square: {}".format(np.unravel_index(chisq.argmin(), chisq.shape)))
        params = util.fit_surface(chisq)
        
        if plot:
            xx,yy = np.meshgrid(range(gridsize), range(gridsize))
            
            if ax == None:
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                ax.plot_wireframe(xx, yy, chisq)
                fig.show()
            else:
                ax.plot_wireframe(xx, yy, chisq)
                return ax
        
        x0,y0 = util.surface_maximum(params)
        d = (self.shape[0] - gridsize) / 2
        logging.debug("\t\tFound star at: {}".format((x0+d, y0+d)))
        return (x0+d, y0+d)

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
        
        star_model_data = np.array([image.star_model_data for image in images])
        self.star_model_data = np.sum(weight_images*star_model_data, axis=0) / np.sum(weight_images, axis=0)
    
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
    