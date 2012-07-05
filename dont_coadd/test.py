# Standard library
import os, sys
import logging
from argparse import ArgumentParser

# Third-party
import matplotlib.cm as cm
import numpy as np
np.set_printoptions(linewidth=150, precision=5)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

import dcimage
import util

def plot_raw_images(images, trial=None):
    """ Generate ... """
    
    for image in images:
        filename = "images/trial{}_index{}.png".format(trial, image.index)
        image.save(filename=filename)        
        # TODO: APW

def plot_coadded_images(images, trial=None):
    """ Generate ... """
    
    min,max = None, None
    for ii,image in enumerate(images):
        filename = "images/trial{}_{}coadded.png".format(trial, ii+1)
        
        if min == None and max == None:
            min,max = image.save(filename=filename)
        else:
            image.save(filename=filename, min=min, max=max)
        
        # TODO: APW

def hogg_test_one(gridsize=3):
    """ This image will be a 3x3 pixel box where each pixel is
        shaded by the chi squared value, labeled with the chi-squared
        value, and a marker over the *true* and *measured* position
        of the star.
    """
    
    positions = []
    for xx in [-0.25, 0.25]:
        for yy in [-0.25, 0.25]:
            positions.append((8.+xx,8.+yy))
    
    for star_position in [(8.,8.)] + positions: 
        for sigma in [0.1, 1.0, 3.0]:
            # == Run first without noise as a proof of concept ==
            image = dcimage.DCImage(shape=(16,16), \
                                  index=0)
            image.add_noise(sky_level=5., \
                          sigma=sigma)
                
            star = dcimage.DCStar(position=star_position, \
                                  flux=64, \
                                  sigma=1.2)
            image.add_star(star=star)
            
            chisq = util.position_chisq(image, gridsize)
            params = util.fit_surface(chisq)
            x0,y0 = util.surface_maximum(params)
            d = (image.shape[0] - gridsize) / 2 + 1
            true_x0, true_y0 = image.star.x0-d, image.star.y0-d
            
            plt.clf()
            plt.subplot(121)
            plt.imshow(image.image_data, cmap=cm.gray, interpolation="none")
            
            plt.subplot(122)
            plt.imshow(chisq, cmap=cm.gray, interpolation="none")
            plt.plot(true_x0, true_y0, 'rD', alpha=0.5, ms=15)
            plt.plot(x0, y0, 'go')
            
            chisq_norm = chisq - chisq.min()
            
            x = y = -0.25
            for ii in range(gridsize):
                xx = x + ii
                for jj in range(gridsize):
                    yy = y + jj
                    plt.text(xx,yy,"{:.2f}".format(chisq_norm[jj,ii]), color="m", fontsize=16)
                    plt.text(xx,yy+0.5,"{:.2f}".format(chisq[jj,ii]), color="b", fontsize=16)
            
            plt.suptitle("Measured position: ({:.5f},{:.5f}) -- True position: ({:.5f},{:.5f})".format(x0,y0,true_x0, true_y0))
            
            plt.savefig("images/tests/hogg_test_one_sigma{:.2f}_star{pos[0]:.2f}-{pos[1]:.2f}.png".format(sigma, pos=star_position))

def best_worst_smoothed_unsmoothed_plot(images, smoothed_images):
    """ Generate two plots: 
        - the first is what we think is the *best* epoch, smoothed and unsmoothed, noisy and noiseless
        - the second is what we think is the *worst* epoch, smoothed and unsmoothed, noisy and noiseless
        
        Note: the Gremlin should f-ck with this! Without the gremlin, for the worst image, the 
            smoothed and unsmoothed should be the same. With the gremlin, this won't be true.
    """
    smoothed_images = [image.smoothed(sigma=worst_sigma) for image in images]
    
    images = sorted(images, key=lambda x: x.star.sigma)
    smoothed_images = sorted(smoothed_images, key=lambda x: x.star.sigma)
    
    plt.figure()
    plt.subplot(221)
    plt.imshow(images[0].star_model_data, cmap=cm.gray, interpolation="none")
    plt.subplot(222)
    plt.imshow(images[0].image_data, cmap=cm.gray, interpolation="none")
    plt.subplot(223)
    plt.imshow(smoothed_images[0].star_model_data, cmap=cm.gray, interpolation="none")
    plt.subplot(224)
    plt.imshow(smoothed_images[0].image_data, cmap=cm.gray, interpolation="none")
    
    plt.figure()
    plt.subplot(221)
    plt.imshow(images[-1].star_model_data, cmap=cm.gray, interpolation="none")
    plt.subplot(222)
    plt.imshow(images[-1].image_data, cmap=cm.gray, interpolation="none")
    plt.subplot(223)
    plt.imshow(smoothed_images[-1].star_model_data, cmap=cm.gray, interpolation="none")
    plt.subplot(224)
    plt.imshow(smoothed_images[-1].image_data, cmap=cm.gray, interpolation="none")
    plt.show()

def plot_grids():
    # TODO: What to do with this code?
    
    # Plot images
    if args.plot and trial == args.plot_trial:
        dcutil.plot_grid(psf_none[0], "images/psfO_noneW.png")
        dcutil.plot_grid(psf_none[1], "images/psfO_noneW_smoothed.png")
        dcutil.plot_grid(psf_sn2[0], "images/psfO_sn2W.png")
        dcutil.plot_grid(psf_sn2[1], "images/psfO_sn2W_smoothed.png")
        
        dcutil.plot_grid(sn2_none[0], "images/sn2O_noneW.png")
        dcutil.plot_grid(sn2_none[1], "images/sn2O_noneW_smoothed.png")
        dcutil.plot_grid(sn2_sn2[0], "images/sn2O_sn2W.png")
        dcutil.plot_grid(sn2_sn2[1], "images/sn2O_sn2W_smoothed.png")
        
        dcutil.plot_grid(psf_none_gremlin[0], "images/psfO_noneW_gremlin.png")
        dcutil.plot_grid(psf_none_gremlin[1], "images/psfO_noneW_smoothed_gremlin.png")
        dcutil.plot_grid(psf_sn2_gremlin[0], "images/psfO_sn2W_gremlin.png")
        dcutil.plot_grid(psf_sn2_gremlin[1], "images/psfO_sn2W_smoothed_gremlin.png")
        
        dcutil.plot_grid(sn2_none_gremlin[0], "images/sn2O_noneW_gremlin.png")
        dcutil.plot_grid(sn2_none_gremlin[1], "images/sn2O_noneW_smoothed_gremlin.png")
        dcutil.plot_grid(sn2_sn2_gremlin[0], "images/sn2O_sn2W_gremlin.png")
        dcutil.plot_grid(sn2_sn2_gremlin[1], "images/sn2O_sn2W_smoothed_gremlin.png")