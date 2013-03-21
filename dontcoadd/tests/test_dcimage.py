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

from ..dcimage import DCStar, DCImage, DCGaussianNoiseModel
from ..util import save_image_data, gaussian

test_path = os.path.split(__file__)[0]

class TestDCStar(object):
    
    def test_addition(self):
        star = DCStar(position=(5,5), flux=10., sigma=2.0)
        image = DCImage((10,10), id="derp")
        
        star_image_l = star + image
        star_image_r = image + star
        
        assert np.all(star_image_l.data == star_image_r.data)

class TestDCGaussianNoiseModel(object):
    
    def test_addition(self):
        nm = DCGaussianNoiseModel(sigma=2.0, sky_level=40.)
        image = DCImage((10,10), id="derp")
        
        star_image_l = nm + image
        star_image_r = image + nm
        
        assert np.all(star_image_l.data == star_image_r.data)
    
class TestDCImage(object):
    
    def test_creation(self):
        image = DCImage(shape=(32,32))
        star = DCStar(position=(31.412, 34.1241), flux=100., sigma=2.0)
        noise = DCGaussianNoiseModel(sky_level=10., sigma=3.0)
        
        image = image + star + noise
    
    def test_star_position(self):
        
        for pos in [(15.,15.), (15.5,15.), (15.,15.5), (15.75, 15.75), (16., 16.)]:
            image = DCImage(shape=(32,32))
            star = DCStar(position=pos, flux=100., sigma=2.0)
            image = image + star
            data = image.data
            
            save_image_data(data, 
                            os.path.join(test_path, "x{0}_y{1}.png".format(pos[0], pos[1])))