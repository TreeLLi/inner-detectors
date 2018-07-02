'''
Linear activation reflector:

To reflect activations of hidden units back to spaces of raw input images

currently supporting linear interpolation

'''


import os, sys
import numpy as np
from easydict import EasyDict as edict

curr_path = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.join(curr_path, "..")
if root_path not in sys.path:
    sys.path.insert(0, root_path)

from src.config import *

def reflect(activ_maps, field_maps, annos):
    print ("linear interpolation")
    for unit_id, unit_activ_maps in activ_maps.items():
        layer = layerOfUnit(unit_id)
        field_map = field_maps[layer]
        input_dims_activs = upsampleL(field_map, unit_activ_maps)            
        anno_dims_activ = [resize(activ, ]
