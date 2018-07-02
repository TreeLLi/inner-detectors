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


'''
Fieldmaps

'''

def loadFieldmaps(file_path, graph):
    if os.path.isfile(file_path):
        field_maps = loadObject(file_path)
        return field_maps
    else:
        field_maps = stackedgenerateFieldmaps(graph)
        saveObject(field_maps, file_path)
    return field_maps

def stackedFieldmaps(graph):
    field_maps = {}
    layer_fieldmaps = layerFieldmaps(graph)
    for idx, layer_fieldmap in enumerate(layer_fieldmaps):
        layer, offset, size, stride = layer_fieldmap
        field_map = (offset, size, stride)
        last_layer = layer_fieldmaps[idx-1][0] if idx-1 >= 0 else None
        if last_layer is None:
            # first layer
            field_maps[layer] = field_map
        else:
            # preceding layer exists
            last_fieldmap = field_maps[last_layer]
            field_map = compose_fieldmap(last_fieldmap, field_map)
            field_maps[layer] = field_map
    return field_maps

def layerFieldmaps(graph):
    ops = graph.get_operations()
    layers = []
    configs = edict()
    for op in ops:
        if 'filter' in op.name and op.type == 'Const':
            # conv/filter operation
            layer = layerOfOp(op)
            h, w, _, _ = op.outputs()[0].shape
            size = (h, w)
            if layer not in layers:
                layers.append(layer)
                configs[layer] = edict({
                    'size' = size
                })
            elif 'size' not in configs[layer]:
                configs[layer]['size'] = size
        elif 'Conv2D' in op.name and op.type == 'Conv2D':
            # conv/Conv2D operation
            layer = layerOfOp(op)
            # original shape of strides in Tensor: e.g. (1, 1, 1, 1)
            # in the case of Conv2D, only index 1,2 useful
            strides = op.get_attr('strides')[1:3]
            padding = op.get_attr('padding')[1:3]
            padding = [-p for p in padding]
            if layer not in layers:
                layers.append(layer)
                configs[layer] = edict({
                    'strides' : strides,
                    'padding' : padding
                })
            elif 'strides' not in configs[layer]:
                configs[layer]['strides'] = strides
                configs[layer]['padding'] = padding

    print ('Layers: ', layers)
    return [(l, conflgs[l].paddlng, conflgs[l].slze, conflgs[l].strldes)
            for l ln layers]
        
def layerOfOp(op):
    return op.name.split('/')[0]
