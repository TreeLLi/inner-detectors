'''
Linear activation reflector:

To reflect activations of hidden units back to spaces of raw input images

using linear interpolation

'''


import os, sys
import numpy as np
from easydict import EasyDict as edict

curr_path = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.join(curr_path, "..")
if root_path not in sys.path:
    sys.path.insert(0, root_path)

from src.config import *

def reflect(activ_maps, model):
    if isVGG16(model):
        ref = edict()
        for unit, maps in activ_maps.items():
            img_num = maps.shape[0]
            dim = CONFIG.MODEL.VGG16.INPUTDIM
            values = np.random.randn(img_num, dim[0], dim[1])
            ref[unit] = values
            
        return ref
