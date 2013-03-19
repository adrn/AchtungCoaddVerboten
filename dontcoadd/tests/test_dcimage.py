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

class TestDCStar(object):
    
    def test_addition(self):
        star = DCStar(flux=10., position=(5,5), sigma=2.0)
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