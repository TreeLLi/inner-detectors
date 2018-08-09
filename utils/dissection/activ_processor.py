'''
Linear activation reflector:

To reflect activations of hidden units back to spaces of raw input images

currently supporting linear interpolation

'''


import os, sys
import numpy as np

from scipy.stats import pearsonr

curr_path = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.join(curr_path, "..")
if root_path not in sys.path:
    sys.path.insert(0, root_path)

from utils.model.model_agent import splitUnitID
from src.config import CONFIG
from utils.dissection.upsample import upsampleL, compose_fieldmap
from utils.dissection.helper import quantile, binarise


'''
Activation maps reflection

'''

def reflect(activ_maps, field_maps, quan=CONFIG.DIS.QUANTILE):
    reflected = {}
    for unit_id, unit_activ_maps in activ_maps.items():
        layer, _ = splitUnitID(unit_id)
        field_map = field_maps[layer]
        # scale activation maps to model input dimensions
        input_dims_activs = upsampleL(field_map, unit_activ_maps)

        quans = quantile(unit_activ_maps, quan, sequence=True)
        binarise(input_dims_activs, quans, sequence=True)
        reflected[unit_id] = input_dims_activs
        
    return reflected


'''
Activation Attributes

'''

ATTRS = ["mean"]
def activAttrs(activ_maps):
    attrs = {}
    for unit_id, unit_activs in activ_maps.items():
        # attribute 0 - mean of activation values
        attrs[unit_id] = [[np.mean(activ)] for activ in unit_activs]
        # TODO - measure more attrs of activation
    return attrs

def correlation(x, y):
    coeff, pvalue = pearsonr(x, y)
    return coeff, pvalue
