'''
Model agent:

To provide a single unified interface to invoke models

'''

import os, sys
import tensorflow as tf
import numpy as np
from easydict import EasyDict as edict

curr_path = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.join(curr_path, "../..")
if root_path not in sys.path:
    sys.path.append(root_path)

from utils.model.vgg16 import Vgg16
from utils.helper.file_manager import loadListFromText, loadObject, saveObject
from utils.dissection.upsample import compose_fieldmap, upsampleL
from src.config import *


'''
Model Agent

'''

class ModelAgent:

    def __init__(self,
                 model=CONFIG.DIS.MODEL,
                 input_size=10,
                 input_dim=CONFIG.MODEL.INPUT_DIM):
        self.field_maps = None
        self.input_dim = (input_size, ) + tuple(input_dim)
        # initialise base model according to the given str
        if isVGG16(model):
            self.model = Vgg16(PATH.MODEL.PARAM)
            self.input_pholder = tf.placeholder("float", self.input_dim)
            self.model.build(self.input_pholder)
            self.graph = self.model.conv1_1.graph
        
    def getActivMaps(self, imgs, layers=None):
        if imgs.shape != self.input_dim:
            # rebuild model for new input dimension
            self.input_dim = imgs.shape
            self.input_pholder = tf.placeholder("float", self.input_dim)
            self.model.build(self.input_pholder)
        feed_dict = {self.input_pholder : imgs}
        activ_maps = activMaps(self.model, feed_dict, layers)
        return activ_maps

    def getFieldmaps(self, file_path=None):
        if self.field_maps is not None:
            return self.field_maps

        # load/generate field maps
        file_path = PATH.MODEL.FIELDMAPS if file_path is None else file_path
        if os.path.isfile(file_path):
            field_maps = loadObject(file_path)
        else:
            field_maps = stackedFieldmaps(self.graph)
            saveObject(field_maps, file_path)
        self.field_maps = field_maps
        return self.field_maps
        
'''
Internal auxiliary functions

'''

def activMaps(model, data, layers):
    if isVGG16(model):
        layers = loadListFromText(PATH.MODEL.PROBE) if layers is None else layers
        with tf.Session() as sess:
            layers_activ_maps = sess.run([getattr(model, layer) for layer in layers],
                                         feed_dict=data)

        activ_maps = edict()
        for idx, layer in enumerate(layers):
            # the dimensions of activation maps: (img_num, width, length, filter_num)
            layer_activ_maps = layers_activ_maps[idx]
            unit_num = layer_activ_maps.shape[3]
            for unit_idx in range(unit_num):
                unit_id = unitOfLayer(layer, unit_idx)
                activ_maps[unit_id] = layer_activ_maps[:, :, :, unit_idx]

        return activ_maps

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
    configs = {}
    for op in ops:
        if 'filter' in op.name and op.type == 'Const':
            # conv/filter operation
            layer = layerOfOp(op)
            h, w, _, _ = op.outputs[0].shape
            size = (int(h), int(w))
            if layer not in layers:
                layers.append(layer)
                configs[layer] = {'size' : size}
            elif 'size' not in configs[layer]:
                configs[layer]['size'] = size
        elif 'Conv2D' in op.name and op.type == 'Conv2D':
            # conv/Conv2D operation
            layer = layerOfOp(op)
            # original shape of strides in Tensor: e.g. (1, 1, 1, 1)
            # in the case of Conv2D, only index 1,2 useful
            strides = tuple(op.get_attr('strides')[1:3])
            padding = op.get_attr('padding')
            padding = _negPaddingFromPadStr(padding, configs[layer]['size'], strides)
            if layer not in layers:
                layers.append(layer)
                configs[layer] = {
                    'strides' : strides,
                    'padding' : padding
                }
            elif 'strides' not in configs[layer]:
                configs[layer]['strides'] = strides
                configs[layer]['padding'] = padding
        elif 'pool' in op.name and 'pool' in op.type.lower():
            # pooling layer
            layer = layerOfOp(op)
            layers.append(layer)
            size = tuple(op.get_attr('ksize')[1:3])
            strides = tuple(op.get_attr('strides')[1:3])
            padding = op.get_attr('padding')
            padding = _negPaddingFromPadStr(padding, size, strides)
            configs[layer] = {
                'size' : size,
                'strides' : strides,
                'padding' : padding
            }

    return [(l, ) + tuple([configs[l][key] for key in ['padding', 'size', 'strides']])
            for l in layers]
    
def _negPaddingFromPadStr(pad_str, ksize, strides):
    # TODO - transform string padding to corresponding numbers
    pad_str = str(pad_str)
    if 'SAME' in pad_str:
        return tuple((s - k) // 2 for k, s in zip(ksize, strides))
    elif 'VALID' in pad_str:
        return (0, 0)

    
'''
ID, Name convertor

'''
    
def unitOfLayer(layer, index):
    return "{}_{}".format(layer, index)

def layerOfUnit(unit_id):
    return unit_id.split('_')[0]

def layerOfOp(op):
    return op.name.split('/')[0]
