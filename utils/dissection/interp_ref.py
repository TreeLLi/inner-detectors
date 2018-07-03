'''
Linear activation reflector:

To reflect activations of hidden units back to spaces of raw input images

currently supporting linear interpolation

'''


import os, sys
import numpy as np
from easydict import EasyDict as edict
from skimage.transformation import resize

curr_path = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.join(curr_path, "..")
if root_path not in sys.path:
    sys.path.insert(0, root_path)

from src.config import *
from utils.model.model_agent import layerOfUnit
from utils.dissection.upsample import upsampleL, compose_fieldmap


'''
Activation maps reflection

'''

def reflect(activ_maps, field_maps, annos):
    for unit_id, unit_activ_maps in activ_maps.items():
        layer = layerOfUnit(unit_id)
        field_map = field_maps[layer]
        # scale activation maps to model input dimensions
        input_dims_activs = upsampleL(field_map, unit_activ_maps)
        # scale activation maps to annotation dimensions
        anno_dims_activs = [resize(activ, annos[i][0].mask.shape)
                           for i, activ in enumerate(input_dims_activs)]
        activ_maps[unit_id] = anno_dims_activs
    return activ_maps
