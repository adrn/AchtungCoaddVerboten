"""

"""

# Standard library
import os, shutil, re
import sys
import copy
from argparse import ArgumentParser
from subprocess import Popen

# Third-party
import sqlalchemy
import numpy as np
np.set_printoptions(threshold=np.nan)
from scipy.stats import scoreatpercentile, chisquare
from scipy.ndimage.filters import gaussian_filter

import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm
import matplotlib.pyplot as plt

import Image
import ImageDraw

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
                Some measure of the spread of the star -- seeing / PSF
        """
        self.x,self.y = position
        self.flux = flux
        self.sigma = sigma


class DCImage(object):
    
    def __init__(self, shape, sky_level, star, sigma=None, trial=None, index=None):
        """ Parameters
            ----------
            shape : tuple
                Should be a tuple of (x,y) values representing the xsize, ysize of the image
            sky_level : float
                The sky brightness for this image
            star : DCStar
                An instantiated DCStar() object
            sigma : float
                The root-variance of the (Gaussian) noise in the image
            trial : int
                The trial number that this image was created
            id : int
                Within a trial, the number the index of the image
        """
                
        # Set default values for missing parameters
        if sigma == None:
            sigma = np.sqrt(np.random.uniform(1.0,2.0))
        
        if index == None:
            index = 0
        
        if trial == None:
            trial = 0
        
        self.shape = shape
        self.size = self.shape[0]*self.shape[1]
        self.star = star
        self.sky_level = sky_level
        self.sigma = sigma
        self.index = index
        self.trial = trial
        
        # Create image arrays
        self.noise_image = np.random.normal(0.0, self.sigma, self.shape)
        self.sky_image = np.zeros(self.shape) + self.sky_level
        
        x1 = np.arange(self.shape[0])
        y1 = np.arange(self.shape[1])
        x,y = np.meshgrid(x1,y1)
        self.star_image = self.star.flux / (2.*np.pi*self.star.sigma**2) * np.exp(- ((x-self.star.x)**2 + (y-self.star.y)**2) / (2*self.star.sigma**2))
    
    @property
    def data(self):
        return self.noise_image + self.sky_image + self.star_image
    
    @property
    def SN2(self):
        return 1.0 / (self.star.sigma**2 * self.sigma**2)
    
    def gremlinize(self, amplitude):
        #self.gremlin_sigma = self.sigma * np.random.uniform(1-amplitude, 1+amplitude)
        #self.gremlin_star_sigma = self.star.sigma * np.random.uniform(1-amplitude, 1+amplitude)
        #self.gremlin_SN2 = 1.0 / (self.gremlin_star_sigma**2 * self.gremlin_sigma**2)
        
        copied = copy.copy(self)
        copied.sigma = self.sigma * np.random.uniform(1-amplitude, 1+amplitude)
        copied.star.sigma = self.star.sigma * np.random.uniform(1-amplitude, 1+amplitude)
        return copied
    
    def show(self):
        plt.imshow(self.data)
        plt.show()
    
    def save(self, path=None, filename=None, min=None, max=None, clip=True):
        if path == None:
            path = os.getcwd()
        
        if not os.path.exists(path):
            raise ValueError("Path '{}' does not exist!".format(path))
        
        if filename == None:
            filename = "trial{}_index{}.png".format(self.trial, self.index)    
        
        return save_image_data(self.data, filename=os.path.join(path,filename), min=min, max=max, clip=clip)

def save_image_data(data, filename, min=None, max=None, clip=True):
    """ Save an array of image data as a png """
    scaled = 255.*linear_scale(data, min=min, max=max, clip=clip)
    
    im = Image.fromarray(scaled.astype(np.uint8))
    im.save(filename)
    
    return min, max

def linear_scale(data, min=None, max=None, clip=True):
    """ Scale an array of image data to values between 0 and 1 """
    
    if min == None:
        min = data.min()
    
    if max == None:
        max = data.max()
    
    scaled = (data - min) / (max - min)
    
    if clip:
        scaled[scaled > 1] = 1.0
        scaled[scaled < 0] = 0.0
    
    return scaled

def cumulative_coadd(images, weight_by=None):
    """ Cumulatively coadd the image data with an optional weight per image """
    
    # We will unravel the images into 1D arrays to make computation faster
    if weight_by == None:
        weight_images = np.ones((len(images), images[0].size))
    else:
        weight_images = np.array([np.ones(images[0].size)*getattr(image, weight_by) for image in images])
    
    images2D = np.array([image.data.ravel()-image.sky_level for image in images])
    coadd_data = np.cumsum(weight_images*images2D, axis=0) / np.cumsum(weight_images, axis=0)
    
    # Get worst PSF to smooth to
    psfs = np.array([image.star.sigma for image in images])
    worst_psf = psfs.max()
    smoothing_sigmas = np.sqrt(worst_psf**2 - psfs**2)
    
    smoothed_images = np.array([gaussian_filter(images2D[ii], smoothing_sigmas[ii]) for ii in range(len(images))])
    coadd_smoothed_data = np.cumsum(weight_images*smoothed_images, axis=0) / np.cumsum(weight_images, axis=0)
    return (coadd_data, coadd_smoothed_data)

def plot_grid(cumcoadded, shape, filename):
    """ Takes an array of 1D images and plots them on a grid
    
    """
    
    #min = max = None
    min = cumcoadded[0].min()
    max = cumcoadded[0].max()
    
    nrows = ncols = int(np.sqrt(len(cumcoadded)))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12,12))
    
    for row in range(nrows):
        for col in range(ncols):
            index = row*ncols + col
            image_data = cumcoadded[index].reshape(shape)
                
            scaled_data = linear_scale(image_data, min=min, max=max)
            axes[row,col].imshow(scaled_data, cmap=cm.gray, interpolation="none")
            #axes[row,col].imshow(image_data, cmap=cm.gray, vmin=min, vmax=max)
            
            axes[row,col].get_xaxis().set_visible(False)
            axes[row,col].get_yaxis().set_visible(False)
            #axes[row,col].set_frame_on(False)
    
    plt.setp([a.get_xticklabels() for a in fig.axes], visible=False)
    plt.setp([a.get_yticklabels() for a in fig.axes], visible=False)
    plt.subplots_adjust(wspace=-0.4, hspace=0.0)
    fig.savefig(filename)

if __name__ == "__main__":
    np.random.seed(42)
    number_of_images = 36
    min,max = None,None
    plot_trial = 0
    
    for trial in range(1):
        images = []
        for index in range(number_of_images):
            star = DCStar((16,16), 4., np.random.uniform(1.0, 1.5))
            im = DCImage(star=star, shape=(32,32), sky_level=np.random.uniform(5.), sigma=np.random.uniform(1.,2.), trial=trial, index=index)
            
            if trial == plot_trial:
                if min == None and max == None:
                    min,max = im.save(path="images")
                else:
                    im.save(path="images", min=min, max=max)
            
            images.append(im)
        
        # PSF ordered:
        images.sort(key=lambda x: x.star.sigma) # IN PLACE
        psf_none = cumulative_coadd(images, weight_by=None)
        psf_sn2 = cumulative_coadd(images, weight_by="SN2")
        
        # SN2 ordered
        images.sort(key=lambda x: x.SN2) # IN PLACE
        sn2_none = cumulative_coadd(images, weight_by=None)
        sn2_sn2 = cumulative_coadd(images, weight_by="SN2")
        
        # PSF ordered:
        gremlin_images = [image.gremlinize(0.1) for image in images]
        gremlin_images.sort(key=lambda x: x.star.sigma) # IN PLACE
        psf_none_gremlin = cumulative_coadd(gremlin_images, weight_by=None)
        psf_sn2_gremlin = cumulative_coadd(gremlin_images, weight_by="SN2")
        
        # SN2 ordered
        images.sort(key=lambda x: x.SN2) # IN PLACE
        sn2_none_gremlin = cumulative_coadd(images, weight_by=None)
        sn2_sn2_gremlin = cumulative_coadd(images, weight_by="SN2")
        
        # Plot images
        if trial == plot_trial:
            plot_grid(psf_none[0], (32,32), "images/psfO_noneW.png")
            plot_grid(psf_none[1], (32,32), "images/psfO_noneW_smoothed.png")
            plot_grid(psf_sn2[0], (32,32), "images/psfO_sn2W.png")
            plot_grid(psf_sn2[1], (32,32), "images/psfO_sn2W_smoothed.png")
            
            plot_grid(sn2_none[0], (32,32), "images/sn2O_noneW.png")
            plot_grid(sn2_none[1], (32,32), "images/sn2O_noneW_smoothed.png")
            plot_grid(sn2_sn2[0], (32,32), "images/sn2O_sn2W.png")
            plot_grid(sn2_sn2[1], (32,32), "images/sn2O_sn2W_smoothed.png")
            
            plot_grid(psf_none_gremlin[0], (32,32), "images/psfO_noneW_gremlin.png")
            plot_grid(psf_none_gremlin[1], (32,32), "images/psfO_noneW_smoothed_gremlin.png")
            plot_grid(psf_sn2_gremlin[0], (32,32), "images/psfO_sn2W_gremlin.png")
            plot_grid(psf_sn2_gremlin[1], (32,32), "images/psfO_sn2W_smoothed_gremlin.png")
            
            plot_grid(sn2_none_gremlin[0], (32,32), "images/sn2O_noneW_gremlin.png")
            plot_grid(sn2_none_gremlin[1], (32,32), "images/sn2O_noneW_smoothed_gremlin.png")
            plot_grid(sn2_sn2_gremlin[0], (32,32), "images/sn2O_sn2W_gremlin.png")
            plot_grid(sn2_sn2_gremlin[1], (32,32), "images/sn2O_sn2W_smoothed_gremlin.png")

## OLD CODE
        
if False:
    parser = ArgumentParser(description="Don't coadd your images!")
    parser.add_argument("-o", "--overwrite", action="store_true", dest="overwrite", default=False,
                    help="Overwrite all pickled data (default = False)")
    parser.add_argument("-v", "--verbose", action="store_true", dest="verbose", default=False,
                    help="Be chatty (default = False)")
    parser.add_argument("-t", "--num-trials", type=int, dest="numTrials", default=1024,
                    help="The number of trials (default = False)")
    parser.add_argument("-s", "--survey", dest="survey", default="panstarrs",
                    help="Controls the number of images to use per trial -- default is panstarrs-like (16 images).")
    parser.add_argument("-z", "--size", type=float, dest="imageSize", default=16,
                    help="The size of the images, default is 16 pixels (square).")
    parser.add_argument("-p", "--plot", action="store_true", dest="plot", default=False,
                    help="Just generate plots with pre-made pickles")
    parser.add_argument("-g", "--gremlin-amplitude", dest="gremlinAmp", default=0.1,
                    help="Some measure of how much your estimates of the star PSF spread and noise sigma are off.")
    
    args = parser.parse_args()
    numTrials = args.numTrials
    overwrite = args.overwrite
    imageSize = args.imageSize
    gremlinAmp = args.gremlinAmp
    survey = args.survey.lower()
    
    if survey.lower() == "lsst":
        numIms = 256
        starFlux = 4.0
    elif survey.lower() == "panstarrs":
        numIms = 16
        starFlux = 32.0
    else:
        raise ValueError("Survey must be either LSST or PANSTARRs")