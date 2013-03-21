# coding: utf-8
"""
    Test the ...
"""

from __future__ import absolute_import, unicode_literals, division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys

# Third-party
import numpy as np
import pytest
import matplotlib.pyplot as plt
from matplotlib import cm

from ..dcimage import DCImage, DCStar, DCGaussianNoiseModel
from ..analyze import position_chisq, _slice_indices, centroid_star
from ..util import save_image_data

test_path = os.path.split(__file__)[0]

class TestPositionChisq(object):
    
    def test_grid_shift(self):
        image = DCImage(shape=(31,31))
        star = DCStar(position=(15.,15.), flux=100., sigma=2.0)
        noise = DCGaussianNoiseModel(sky_level=0., sigma=0.001)
        image = image + star + noise
        
        gridsize = 3
        
        # This code should be the same as in position_chisq()
        
        shp = (gridsize,gridsize)
        xx,yy = _slice_indices(image, gridsize)
        
        # Empty array to fill with chisq values evaluated for each offset
        chisq = np.zeros(shp, dtype=float)
        
        data_cutout = image.data[yy[0]:yy[1], xx[0]:xx[1]]
        
        # Sky subtraction: we assume know the sky level because we have plenty of
        #   image to model this well
        data_cutout -= image.sky_level
        
        # Create a model image -- the star by itself
        star_model = image.star.as_image(image.shape)
        
        min = None
        max = None
        for ii in range(gridsize):
            for jj in range(gridsize):
                y_offset = ii-int(np.floor(gridsize / 2))
                x_offset = jj-int(np.floor(gridsize / 2))
                xidx,yidx = _slice_indices(image, gridsize, x_offset, y_offset)
                
                star_cutout = star_model[yidx[0]:yidx[1], xidx[0]:xidx[1]]
                chisq[ii,jj] = np.sum((data_cutout-star_cutout)**2) / image.sigma**2
                
                filename = os.path.join(test_path, "grid_shift_ii{0}_jj{1}.png".format(ii,jj))
                if min == None:
                    min = star_cutout.min()
                if max == None:
                    max = star_cutout.max()
                save_image_data(star_cutout, filename, min=min, max=max, clip=True)
        
    @pytest.mark.parametrize(("true_x0","true_y0"), [(15.,15.),(15.5,15.),(15.3,15.8),(16.,16.)])
    def test_centroid_star(self, true_x0, true_y0):
        image = DCImage(shape=(31,31))
        star = DCStar(position=(true_x0,true_y0), flux=100., sigma=2.0)
        noise = DCGaussianNoiseModel(sky_level=0., sigma=0.001)
        image = image + star + noise
        
        gridsize = 3
        chisq = position_chisq(image, gridsize=gridsize)
        x0,y0 = centroid_star(image, gridsize=gridsize)
        offset = np.sqrt((x0-image.star.x0)**2+(y0-image.star.y0)**2)
        
        xx,yy = _slice_indices(image, gridsize, x_offset=0, y_offset=0)

        print ("Offset: {0}".format(offset))
        #assert offset < 1. # crude test...
        
        fig,ax = plt.subplots(1,1,figsize=(8,8))
        ax.imshow(np.flipud(chisq), cmap=cm.Greys, interpolation="nearest",
                  extent=[xx[0]-0.5,xx[1]-0.5,yy[0]-0.5,yy[1]-0.5])
        ax.plot([star.x0],[star.y0], color="r", 
                marker="o", markeredgecolor="none", markersize=5, 
                label="true", alpha=0.6)
        ax.plot([x0],[y0], color="b", 
                marker="o", markeredgecolor="none", markersize=5,
                label="measured", alpha=0.6)
        ax.legend()
        
        filename = os.path.join(test_path, "centroid_test_grid_x{0}_y{1}.png".format(true_x0,true_y0))
        fig.savefig(filename)
    
    def test_half_integer_offset(self):
        image = DCImage(shape=(31,31))
        star = DCStar(position=(15.8,15.3), flux=100., sigma=2.0)
        #noise = DCGaussianNoiseModel(sky_level=0., sigma=0.001)
        image = image + star# + noise
        
        gridsize = 3
        
        shp = (gridsize,gridsize)
        xx,yy = _slice_indices(image, gridsize)
        # Create a model image -- the star by itself
        star_cutout = image.star.as_image(image.shape)[yy[0]:yy[1], xx[0]:xx[1]]
        
        # Empty array to fill with chisq values evaluated for each offset
        chisq = np.zeros(shp, dtype=float)
        
        for ii in range(gridsize):
            for jj in range(gridsize):
                y_offset = ii-int(np.floor(gridsize / 2))
                x_offset = jj-int(np.floor(gridsize / 2))
                xidx, yidx = _slice_indices(image, gridsize, x_offset, y_offset)
                
                #star_cutout = star_model[yidx[0]:yidx[1], xidx[0]:xidx[1]]
                data_cutout = image.data[yidx[0]:yidx[1], xidx[0]:xidx[1]]
                
                # Sky subtraction: we assume know the sky level because we have plenty of
                #   image to model this well
                data_cutout -= image.sky_level
                
                chisq[jj,ii] = np.sum((data_cutout-star_cutout)**2)
                
                fig,axes = plt.subplots(1,2,figsize=(8,8))
                fig.suptitle(chisq[jj,ii])
                
                padded_data = np.zeros((5,5))
                padded_data[y_offset+1:y_offset+4,x_offset+1:x_offset+4] = data_cutout
                axes[0].imshow(np.flipud(padded_data), cmap=cm.Greys_r, interpolation="nearest",
                               extent=[xx[0]-1.5,xx[1]+0.5,yy[0]-1.5,yy[1]+0.5], 
                               vmin=star_cutout.min(),vmax=star_cutout.max())
                axes[0].plot([star.x0],[star.y0], color="r", 
                        marker="o", markeredgecolor="none", markersize=5, 
                        label="true", alpha=0.6)
                axes[0].set_title("data")
                
                axes[1].imshow(np.flipud(star_cutout), cmap=cm.Greys_r, interpolation="nearest",
                               extent=[xx[0]-0.5,xx[1]-0.5,yy[0]-0.5,yy[1]-0.5])
                axes[1].plot([star.x0],[star.y0], color="r", 
                        marker="o", markeredgecolor="none", markersize=5, 
                        label="true", alpha=0.6)
                axes[1].set_title("model")
                        
                filename = os.path.join(test_path, "half_integer_xoffset{0}_yoffset{1}.png".format(x_offset,y_offset))
                fig.savefig(filename)
        
        fig,ax = plt.subplots(1,1,figsize=(8,8))
        ax.imshow(-np.flipud(chisq), cmap=cm.Greys_r, interpolation="nearest",
                  extent=[-1.5,1.5,-1.5,1.5])
        
        for ii in range(gridsize):
            for jj in range(gridsize):
                ax.text(ii-1,jj-1, "{0}".format(chisq[jj,ii]), color="red")
        fig.savefig(os.path.join(test_path, "half_integer_chisq.png"))
        
        
        
        