from __future__ import division

""" """

# Standard library
import os, sys
import logging
from argparse import ArgumentParser

# Third-party
import matplotlib
matplotlib.use("WxAgg")
import numpy as np
np.set_printoptions(linewidth=150, precision=5)
import matplotlib.pyplot as plt

# Project
import dont_coadd.util as dcu
import dont_coadd.dcimage as dcimage

if __name__ == "__main__":
    parser = ArgumentParser(description="Don't coadd your images!")
    #parser.add_argument("-o", "--overwrite", action="store_true", dest="overwrite", default=False,
    #                help="Overwrite all pickled data (default = False)")
    parser.add_argument("-v", "--verbose", action="store_true", dest="verbose", default=False,
                    help="Be chatty (default = False)")
    parser.add_argument("--num-trials", type=int, dest="num_trials", default=1024,
                    help="The number of trials (default = False)")
    parser.add_argument("--num-per-trial", dest="num_per_trial", required=True, type=int,
                    help="Controls the number of images to use per trial.")
    parser.add_argument("--image-size", type=int, dest="image_size", default=16,
                    help="The size of the images, default is 16 pixels (square).")
    parser.add_argument("-p", "--plot", action="store_true", dest="plot", default=False,
                    help="Generate any plots")
    parser.add_argument("--plot-trial", type=int, dest="plot_trial", default=0,
                    help="Generate plots for given trial.")
    parser.add_argument("--gremlin-amplitude", dest="gremlin_amplitude", default=0.1, type=float,
                    help="Some measure of how much your estimates of the star PSF spread and noise sigma are off.")
    parser.add_argument("--seed", dest="seed", default=42,
                    help="Set the random number generator seed.")
    parser.add_argument("--star-flux", dest="star_flux", default=8., type=float,
                    help="The flux of the star to add to each image.")
    
    args = parser.parse_args()
    np.random.seed(args.seed)
    
    if args.verbose: logging.basicConfig(level=logging.DEBUG)
    else: logging.basicConfig(level=logging.INFO)
    
    # Enforce an even number of pixels
    if args.image_size % 2 != 0:
        raise ValueError("Image size must be an even integer. You entered: {}".format(args.image_size))
    
    xs = []
    for trial in range(args.num_trials):    
        images = []
        min,max = None,None # used for stretching individual images
        logging.debug("Trial {}".format(trial))
        for index in range(args.num_per_trial):
            logging.debug("\tIndex {}".format(index))
            
            # Create a star with uniform random spread
            star_pos = args.image_size // 2
            star_sigma = np.random.uniform(1.5, 2.0)
            logging.debug("\t\t- Star sigma: {:0.2f}".format(star_sigma))
            star = dcimage.DCStar((star_pos,star_pos), args.star_flux, star_sigma)
            
            # Create an image, add star to image and add read noise / sky
            sky_level = np.random.uniform(5.) # APW: units?
            logging.debug("\t\t- Sky level: {:0.2f}".format(sky_level))
            noise_sigma = np.random.uniform(1, 1.5)
            logging.debug("\t\t- Read noise sigma: {:0.2f}".format(noise_sigma))
            
            im = dcimage.DCImage(shape=(args.image_size,args.image_size), index=index)
            im.add_star(star=star)
            im.add_noise(sky_level=sky_level, sigma=noise_sigma)
            
            if args.plot and trial == args.plot_trial:
                filename = "images/trial{}_index{}.png".format(trial, index)
                if min == None and max == None:
                    min,max = im.save(filename=filename)
                else:
                    im.save(filename=filename, min=min, max=max)
            
            images.append(im)
        
        # Get worst sigma to smooth to
        star_sigmas = np.array([image.star.sigma for image in images])
        worst_psf = star_sigmas.max()
        smoothed_images = [image.smoothed(sigma=worst_psf) for image in images]
        
        # ==========================================================
        # Sort on INCREASING PSF width:
        images.sort(key=lambda x: x.star.sigma) # SORT IN PLACE
        smoothed_images.sort(key=lambda x: x.star.sigma) # SORT IN PLACE
        
        #   - Not Weighted
        psf_none = dcimage.cumulative_coadd(images, weight_by=None)
        psf_none_smoothed = dcimage.cumulative_coadd(smoothed_images, weight_by=None)
        
        #   - Weighted by [Signal/Noise]^2
        psf_sn2 = dcimage.cumulative_coadd(images, weight_by="SN2")
        psf_sn2_smoothed = dcimage.cumulative_coadd(smoothed_images, weight_by="SN2")
        
        if args.verbose and args.plot and trial == args.plot_trial:
            fig = plt.figure()
            ax1 = fig.add_subplot(221)
            psf_none[-1].show(ax1)
            ax2 = fig.add_subplot(222)
            psf_none_smoothed[-1].show(ax2)
            ax3 = fig.add_subplot(223)
            psf_sn2[-1].show(ax3)
            ax4 = fig.add_subplot(224)
            psf_sn2_smoothed[-1].show(ax4)
            plt.show()
        
        # ==========================================================
        # Sort on DECREASING [Signal/Noise]^2:
        images.sort(key=lambda x: x.SN2, reverse=True) # SORT IN PLACE
        smoothed_images.sort(key=lambda x: x.SN2, reverse=True) # SORT IN PLACE
        
        #   - Not Weighted
        sn2_none = dcimage.cumulative_coadd(images, weight_by=None)
        sn2_none_smoothed = dcimage.cumulative_coadd(smoothed_images, weight_by=None)
        
        #   - Weighted by [Signal/Noise]^2
        sn2_sn2 = dcimage.cumulative_coadd(images, weight_by="SN2")
        sn2_sn2_smoothed = dcimage.cumulative_coadd(smoothed_images, weight_by="SN2")
        
        # APW: Skip gremlin for now
        continue
        
        # ==========================================================
        # GREMLIN!
        # PSF ordered:
        gremlin_images = [image.gremlinized(0.1) for image in images]
        gremlin_images.sort(key=lambda x: x.star.sigma) # SORT IN PLACE
        psf_none_gremlin = dcimage.cumulative_coadd(gremlin_images, weight_by=None)
        psf_sn2_gremlin = dcimage.cumulative_coadd(gremlin_images, weight_by="SN2")
        
        # SN2 ordered
        images.sort(key=lambda x: x.SN2, reverse=True) # SORT IN PLACE
        sn2_none_gremlin = dcimage.cumulative_coadd(images, weight_by=None)
        sn2_sn2_gremlin = dcimage.cumulative_coadd(images, weight_by="SN2")
        
        # Plot images
        if args.plot and trial == args.plot_trial:
            dcu.plot_grid(psf_none[0], "images/psfO_noneW.png")
            dcu.plot_grid(psf_none[1], "images/psfO_noneW_smoothed.png")
            dcu.plot_grid(psf_sn2[0], "images/psfO_sn2W.png")
            dcu.plot_grid(psf_sn2[1], "images/psfO_sn2W_smoothed.png")
            
            dcu.plot_grid(sn2_none[0], "images/sn2O_noneW.png")
            dcu.plot_grid(sn2_none[1], "images/sn2O_noneW_smoothed.png")
            dcu.plot_grid(sn2_sn2[0], "images/sn2O_sn2W.png")
            dcu.plot_grid(sn2_sn2[1], "images/sn2O_sn2W_smoothed.png")
            
            dcu.plot_grid(psf_none_gremlin[0], "images/psfO_noneW_gremlin.png")
            dcu.plot_grid(psf_none_gremlin[1], "images/psfO_noneW_smoothed_gremlin.png")
            dcu.plot_grid(psf_sn2_gremlin[0], "images/psfO_sn2W_gremlin.png")
            dcu.plot_grid(psf_sn2_gremlin[1], "images/psfO_sn2W_smoothed_gremlin.png")
            
            dcu.plot_grid(sn2_none_gremlin[0], "images/sn2O_noneW_gremlin.png")
            dcu.plot_grid(sn2_none_gremlin[1], "images/sn2O_noneW_smoothed_gremlin.png")
            dcu.plot_grid(sn2_sn2_gremlin[0], "images/sn2O_sn2W_gremlin.png")
            dcu.plot_grid(sn2_sn2_gremlin[1], "images/sn2O_sn2W_smoothed_gremlin.png")