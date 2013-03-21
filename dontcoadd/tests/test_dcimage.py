# coding: utf-8
"""
    Test the classes from dcimage.py
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

from ..dcimage import DCStar, DCImage, DCGaussianNoiseModel, DCCoaddmaschine
from ..util import save_image_data, gaussian

test_path = os.path.split(__file__)[0]

class TestDCStar(object):
    
    def test_addition(self):
        star = DCStar(position=(5,5), flux=10., sigma=2.0, shape=(10,10))
        image = DCImage((10,10), id="derp")
        
        star_image_l = star + image
        star_image_r = image + star
        
        assert np.all(star_image_l.data == star_image_r.data)

class TestDCGaussianNoiseModel(object):
    
    def test_addition(self):
        nm = DCGaussianNoiseModel(sigma=2.0, sky_level=40., shape=(10,10))
        image = DCImage((10,10), id="derp")
        
        star_image_l = nm + image
        star_image_r = image + nm
        
        assert np.all(star_image_l.data == star_image_r.data)
    
class TestDCImage(object):
    
    def test_creation(self):
        image = DCImage(shape=(32,32))
        star = DCStar(position=(31.412, 34.1241), flux=100., sigma=2.0, shape=(32,32))
        noise = DCGaussianNoiseModel(sky_level=10., sigma=3.0, shape=(32,32))
        
        image = image + star + noise
    
    def test_star_position(self):
        
        for pos in [(15.,15.), (15.5,15.), (15.,15.5), (15.75, 15.75), (16., 16.)]:
            image = DCImage(shape=(32,32))
            star = DCStar(position=pos, flux=100., sigma=2.0, shape=(32,32))
            image = image + star
            data = image.data
            
            save_image_data(data, 
                            os.path.join(test_path, "x{0}_y{1}.png".format(pos[0], pos[1])))

class TestDCCoaddmaschine(object):
    
    def test_basic(self):
        
        images = []
        for ii in range(32):
            image = DCImage((16,16), id=ii)
            star = DCStar(position=(8.,8.), 
                          flux=64., 
                          sigma=np.random.uniform(1.,2.), 
                          shape=image.shape)
            noise = DCGaussianNoiseModel(sky_level=np.random.uniform(0,10),
                                         sigma=np.random.uniform(1.,4.),
                                         shape=image.shape)
            images.append(image+noise+star)
        
        for image in images:
            image.save(os.path.join(test_path,"{0}.png".format(image.id)))
        
        maschine = DCCoaddmaschine(images)
        coadded_images = maschine.coadd(weight_by=None, sort_by="sn2")

        for image in coadded_images:
            image.save(os.path.join(test_path,"coadded_{0}.png".format(image.id)))
        
    def test_sort_grids(self):
    
        images = []
        for ii in range(32):
            image = DCImage((16,16), id=ii)
            star = DCStar(position=(8.,8.), 
                          flux=64., 
                          sigma=np.random.uniform(1.,3.), 
                          shape=image.shape)
            noise = DCGaussianNoiseModel(sky_level=np.random.uniform(0,100),
                                         sigma=np.random.uniform(1.,4.),
                                         shape=image.shape)
            images.append(image+noise+star)
        
        maschine = DCCoaddmaschine(images)
        
        d = np.ravel([image.data.ravel() for image in images])
        vmin, vmax = d.min(), d.max()
        
        sn2_sorted = maschine.sorted_images(sort_by="sn2")
        fig,axes = plot_image_grid(sn2_sorted, vmin=vmin, vmax=vmax)
        for ii,ax in enumerate(np.ravel(axes)):
            print(sn2_sorted[ii].SN2)
            ax.text(3,3,
                    r"$\sigma_s$={0:.2f}".format(sn2_sorted[ii].star.sigma),
                    color="red")
            ax.text(3,5,
                    r"$[S/N]^2$={0:.2f}".format(sn2_sorted[ii].SN2),
                    color="red")
        fig.savefig(os.path.join(test_path, "sn2_sorted.png"))
        #maschine.sorted_images(sort_by="psf")
        
        vmin,vmax = None,None
        for weight_by in [None, "sn2"]:
            for sort_by in ["sn2", "psf"]:
                coadded_images = maschine.coadd(weight_by=weight_by, sort_by=sort_by)
                if vmin == None:
                    vmin,vmax = coadded_images[-1].data.min(), coadded_images[-1].data.max()
                plot_image_grid(coadded_images, vmin=vmin, vmax=vmax, 
                                filename=os.path.join(test_path, "grid_sort{0}_weight{1}.png".format(sort_by, weight_by)))
                
def plot_image_grid(images, vmin=None, vmax=None, filename=None):
    size = int(np.sqrt(len(images)))
    fig,axes = plt.subplots(size,size,figsize=(16,16))
    
    if vmin == None:
        vmin,vmax = images[-1].data.min(), coadded_images[-1].data.max()
        
    for ii,ax in enumerate(np.ravel(axes)):
        ax.imshow(images[ii].data, vmin=vmin, vmax=vmax,
                  cmap=cm.Greys_r, interpolation="none")
    
    if filename == None:
        return fig,axes
    else:
        fig.savefig(filename)
        return None