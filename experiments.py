# coding: utf-8
from __future__ import division

""" """

# Standard library
import os, sys
import logging
from argparse import ArgumentParser

# Third-party
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.cm as cm
import numpy as np
np.set_printoptions(linewidth=150, precision=5)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

# Project
import dont_coadd.util as dcutil
import dont_coadd.dcimage as dcimage
import dont_coadd.test as dctest

# ==================================================================================================

def compute_offsets(coadded_images):
    # TODO: Best way to pass plot keywords?
    
    offsets = []
    for coadded_image in coadded_images:
        x0,y0 = coadded_image.centroid_star(gridsize=5)
        offsets.append(np.sqrt((x0-coadded_image.star.x0)**2 + (y0-coadded_image.star.y0)**2))
    
    return np.array(offsets)
    
def test_mode():
    dctest.hogg_test_one(gridsize=3)
    sys.exit(0)

if __name__ == "__main__":
    parser = ArgumentParser(description="Don't coadd your images!")
    #parser.add_argument("-o", "--overwrite", action="store_true", dest="overwrite", default=False,
    #                help="Overwrite all pickled data (default = False)")
    parser.add_argument("-v", "--verbose", dest="verbose", action='count', default=0,
                    help="Be chatty (default = False)")
    parser.add_argument("--num-trials", type=int, dest="num_trials", default=1024,
                    help="The number of trials (default = False)")
    parser.add_argument("--images-per-trial", dest="images_per_trial", type=int,
                    help="Controls the number of images to use per trial.")
    parser.add_argument("--image-size", type=int, dest="image_size", default=16,
                    help="The size of the images, default is 16 pixels (square).")
    parser.add_argument("-p", "--plot", action="store_true", dest="plot", default=False,
                    help="Generate any plots")
    parser.add_argument("--plot-trial", type=int, dest="plot_trial", default=0,
                    help="Generate plots for given trial.")
    parser.add_argument("--gremlin", action="store_true", dest="gremlin", default=False,
                    help="Gremlinize our knowledge!")
    parser.add_argument("--gremlin-amplitude", dest="gremlin_amplitude", default=0.1, type=float,
                    help="Some measure of how much your estimates of the star PSF spread and noise sigma are off.")
    parser.add_argument("--seed", dest="seed", default=42, type=int,
                    help="Set the random number generator seed.")
    parser.add_argument("--star-flux", dest="star_flux", default=8., type=float,
                    help="The flux of the star to add to each image.")
    parser.add_argument("--test", action="store_true", dest="test", default=False,
                    help="Run in test mode.")
    
    args = parser.parse_args()
    np.random.seed(args.seed)
    
    # Create logger
    logger = logging.getLogger(__name__)
    ch = logging.StreamHandler() # console handler
    ch.setLevel(logging.ERROR)
    
    formatter = logging.Formatter("%(name)s / %(levelname)s / %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    logger.setLevel(logging.INFO)
    dcimage.logger.setLevel(logging.INFO)
    
    if args.verbose > 0:
        logger.setLevel(logging.DEBUG)
    if args.verbose > 1:
        dcimage.logger.setLevel(logging.DEBUG)
    
    if args.test:
        test_mode()
    
    # Enforce an even number of pixels
    if args.image_size % 2 != 0:
        raise ValueError("Image size must be an even integer. You entered: {}".format(args.image_size))
    
    psf_none_offsets = np.zeros((args.num_trials,args.images_per_trial), dtype=float)
    psf_none_smoothed_offsets = np.zeros((args.num_trials,args.images_per_trial), dtype=float)
    psf_sn2_offsets = np.zeros((args.num_trials,args.images_per_trial), dtype=float)
    psf_sn2_smoothed_offsets = np.zeros((args.num_trials,args.images_per_trial), dtype=float) 
    sn2_none_offsets = np.zeros((args.num_trials,args.images_per_trial), dtype=float)
    sn2_none_smoothed_offsets = np.zeros((args.num_trials,args.images_per_trial), dtype=float)
    sn2_sn2_offsets = np.zeros((args.num_trials,args.images_per_trial), dtype=float)
    sn2_sn2_smoothed_offsets = np.zeros((args.num_trials,args.images_per_trial), dtype=float)
    
    # Start the main experiment
    for trial in range(args.num_trials):
        logger.info("Trial {}".format(trial))
        
        # Empty list to store all of the image objects for a single trial
        images = []
        
        # Compute star position for all images in this trial
        #star_position = np.random.uniform(-1,1,size=2) + args.image_size//2
        star_position = (8., 8.)
        
        # Create images to be sorted and coadded. The number depends on the type of 'survey'
        #   e.g. should be ~32 for Pan-STARRS, ~256 for LSST
        for index in range(args.images_per_trial):
            logger.debug("Index {}".format(index))
            
            # Create an image, add star to image and add read noise / sky
            img = dcimage.DCImage(shape=(args.image_size,args.image_size), \
                                  index=index)
            #img.add_noise(sky_level=np.random.uniform(5.), \
            #              sigma=np.random.uniform(1, 1.5)) # APW: MAGIC NUMBERS!
            img.add_noise(sky_level=np.random.uniform(5.), \
                          sigma=0.1) # APW: MAGIC NUMBERS!
            
            # Create a star with a FWHM drawn from a uniform distribution and a position drawn
            #   from a distribution within a few pixels of the center of the image
            star = dcimage.DCStar(position=star_position, \
                                  flux=args.star_flux, \
                                  sigma=np.random.uniform(1., 2.)) # APW: MAGIC NUMBERS!
            img.add_star(star=star)
            images.append(img)
        
        if args.gremlin:
            logger.debug("Gremlinizing images!")
            images = [image.gremlinized(0.1) for image in images]
        
        if args.plot and trial == args.plot_trial:
            dctest.plot_raw_images(images)
        
        dctest.hogg_test_one(images[0])
        sys.exit(0)
        
        # Get the worst seeing image (sigma) to smooth the other epochs
        star_sigmas = np.array([image.star.sigma for image in images])
        worst_sigma = star_sigmas.max()
        dcimage.logger.debug("Worst σ = {:.2f}, Best σ = {:.2f}".format(worst_sigma, star_sigmas.min()))
        
        if args.verbose > 2 and args.plot and trial == args.plot_trial:
            test.best_worst_smoothed_unsmoothed_plot(images)
        
        # All of the below should get wrapped in a function that takes a list of images
        smoothed_images = [image.smoothed(sigma=worst_sigma) for image in images]
        
        # ==========================================================================================
        # Sort on INCREASING PSF FWHM:
        #   e.g. image 0 has the *best* seeing, image [-1] has the *worst* after sort
        images.sort(key=lambda x: x.star.sigma) # SORT IN PLACE
        smoothed_images.sort(key=lambda x: x.star.sigma) # SORT IN PLACE
        
        # - Not Weighted
        psf_none = dcimage.cumulative_coadd(images, weight_by=None)
        psf_none_smoothed = dcimage.cumulative_coadd(smoothed_images, weight_by=None)
        psf_none_offsets[trial] = compute_offsets(psf_none)
        psf_none_smoothed_offsets[trial] = compute_offsets(psf_none_smoothed)
        
        # - Weighted by [Signal/Noise]^2
        psf_sn2 = dcimage.cumulative_coadd(images, weight_by="SN2")
        psf_sn2_smoothed = dcimage.cumulative_coadd(smoothed_images, weight_by="SN2")
        psf_sn2_offsets[trial] = compute_offsets(psf_sn2)
        psf_sn2_smoothed_offsets[trial] = compute_offsets(psf_sn2_smoothed)
        
        # ==========================================================================================
        # Sort on DECREASING [Signal/Noise]^2:
        images.sort(key=lambda x: x.SN2, reverse=True) # SORT IN PLACE
        smoothed_images.sort(key=lambda x: x.SN2, reverse=True) # SORT IN PLACE
        
        # - Not Weighted
        sn2_none = dcimage.cumulative_coadd(images, weight_by=None)
        sn2_none_smoothed = dcimage.cumulative_coadd(smoothed_images, weight_by=None)
        sn2_none_offsets[trial] = compute_offsets(sn2_none)
        sn2_none_smoothed_offsets[trial] = compute_offsets(sn2_none_smoothed)
        
        # - Weighted by [Signal/Noise]^2
        sn2_sn2 = dcimage.cumulative_coadd(images, weight_by="SN2")
        sn2_sn2_smoothed = dcimage.cumulative_coadd(smoothed_images, weight_by="SN2")
        sn2_sn2_offsets[trial] = compute_offsets(sn2_sn2)
        sn2_sn2_smoothed_offsets[trial] = compute_offsets(sn2_sn2_smoothed)
    
    
    
    
    
    markers = ["o", ".", "^", "*", "D", "s", "d", "h"]
    
    offset_arrays = [psf_none_offsets, \
                     psf_none_smoothed_offsets, \
                     psf_sn2_offsets, \
                     psf_sn2_smoothed_offsets, \
                     sn2_none_offsets, \
                     sn2_none_smoothed_offsets, \
                     sn2_sn2_offsets, \
                     sn2_sn2_smoothed_offsets]
    
    offset_array_keys = ["psf_none_offsets", \
                        "psf_none_smoothed_offsets", \
                        "psf_sn2_offsets", \
                        "psf_sn2_smoothed_offsets", \
                        "sn2_none_offsets", \
                        "sn2_none_smoothed_offsets", \
                        "sn2_sn2_offsets", \
                        "sn2_sn2_smoothed_offsets"]
    
    for ii in range(len(offset_arrays)):
        a = np.median(offset_arrays[ii], axis=0)
        plt.plot(range(len(a)), a, label=offset_array_keys[ii], linestyle="none", marker=markers[ii], alpha=0.65, ms=10)
    
    plt.ylim(0,1)
    plt.legend()
    plt.show()
