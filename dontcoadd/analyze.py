# coding: utf-8

""" Description... """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys

# Third-party
import numpy as np

from .util import fit_quad_surface, quad_surface_maximum
from .dcimage import DCImage, DCStar

def _slice_indices(image, gridsize, x_offset=0, y_offset=0):
    """ Given an image with a star and a gridsize, figure out what the slice
        indices are to get a grid (of size gridsize) around the star.
    """
    
    g = int(np.floor(gridsize / 2))
    x_idx1 = int(np.round(image.star.x0)) - g + x_offset
    x_idx2 = int(np.round(image.star.x0)) + g + 1 + x_offset
    
    y_idx1 = int(np.round(image.star.y0)) - g + y_offset
    y_idx2 = int(np.round(image.star.y0)) + g + 1 + y_offset
    
    if x_idx1 < 0: x_idx1 = 0
    if y_idx1 < 0: y_idx1 = 0
    if x_idx2 > image.shape[0]-1: x_idx2 = image.shape[0]
    if y_idx2 > image.shape[1]-1: y_idx2 = image.shape[1]
    
    return (x_idx1, x_idx2), (y_idx1, y_idx2)

def position_chisq(image, model_image=None, gridsize=3):
    """ Compute the chisquared value for placing the model in every position
        on a 3x3 grid around the aligned positions.
    """
    
    shp = (gridsize,gridsize)
    xx,yy = _slice_indices(image, gridsize)
    data_cutout = image.data[yy[0]:yy[1], xx[0]:xx[1]]
    
    # Empty array to fill with chisq values evaluated for each offset
    chisq = np.zeros(shp, dtype=float)
    
    # Sky subtraction: we assume know the sky level because we have plenty of
    #   image to model this well
    data_cutout -= image.sky_level
    
    # Create a model image -- the star by itself
    if model_image == None:
        star_model = image.star.data
    else:
        star_model = model_image.data
    
    for ii in range(gridsize):
        for jj in range(gridsize):
            y_offset = ii-int(np.floor(gridsize / 2))
            x_offset = jj-int(np.floor(gridsize / 2))
            xidx, yidx = _slice_indices(image, gridsize, x_offset, y_offset)
            
            star_cutout = star_model[yidx[0]:yidx[1], xidx[0]:xidx[1]]
            chisq[ii,jj] = np.sum((data_cutout-star_cutout)**2) / image.sigma**2

    return chisq

def centroid_star(image, model_image=None, gridsize=3):
    """ Given noisy image data, and the 'model' image data (noiseless image),
        compute the chi-squared at all positions on a 3x3 grid around the nominal
        (true, known) position of the star.
    """
    
    chisq = position_chisq(image, model_image, gridsize=gridsize)
    params = fit_quad_surface(chisq)
    
    x0,y0 = quad_surface_maximum(params)
    xx,yy = _slice_indices(image, gridsize, x_offset=0, y_offset=0)
    
    return (x0+xx[0], y0+yy[0])