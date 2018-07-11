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
                 input_dim=CONFIG.MODEL.INPUT_DIM,
                 deconv=False):
        self.field_maps = None
        self.input_dim = (input_size, ) + tuple(input_dim)
        self.deconv = deconv
        
        # initialise base model according to the given str
        if isVGG16(model):
            self.model = Vgg16(PATH.MODEL.CONFIG, PATH.MODEL.PARAM, deconv=deconv)
            self.input_pholder = tf.placeholder("float", self.input_dim)
            self.model.build(self.input_pholder)

    def getFieldmaps(self, file_path=None):
        if self.field_maps is not None:
            return self.field_maps

        # load/generate field maps
        file_path = PATH.MODEL.FIELDMAPS if file_path is None else file_path
        if os.path.isfile(file_path):
            print ("Fieldmaps: loading from the stored object file ...")
            field_maps = loadObject(file_path)
        else:
            print ("Fieldmaps: generating ...")
            field_maps = stackedFieldmaps(self.model)
            saveObject(field_maps, file_path)
            print ("Fieldmaps: saved at {}".format(file_path))
        self.field_maps = field_maps
        return self.field_maps
            
    def getActivMaps(self, imgs, layers):
        print ("Fetching activation maps for specific units ...")
        # generate activation data by forward pass
        if imgs.shape != self.input_dim:
            # rebuild model for new input dimension
            self.input_dim = imgs.shape
            self.input_pholder = tf.placeholder("float", self.input_dim)
            self.model.build(self.input_pholder)

        fetches = {"activs" : {}}
        for layer in layers:
            fetches['activs'][layer] = self.model.getLayerTensor(layer)
        if self.deconv:
            fetches['switches'] = self.model.max_pool_switches
        feed_dict = {self.input_pholder : imgs}
        with tf.Session() as sess:
            results = sess.run(fetches, feed_dict=feed_dict)

        activ_maps = edict()
        for layer, layer_activ_maps in results['activs'].items():
            # the dimensions of activation maps: (img_num, width, length, filter_num)
            unit_num = layer_activ_maps.shape[3]
            for unit_idx in range(unit_num):
                unit_id = unitOfLayer(layer, unit_idx)
                activ_maps[unit_id] = layer_activ_maps[:, :, :, unit_idx]

        if not self.deconv:
            return activ_maps
        else:
            return activ_maps, results['switches']

    def getDeconvMaps(self, activ_maps, switches):
        feed_dict = {self.model.getSwitchTensor(layer) : switch
                     for layer, switch in switches.items()}
        
        for unit_id, activ_map in activ_maps.items():
            layer, unit = splitUnitID(unit_id)
            config = self.model.getConfig(layer)
            ksize = config.ksize
            activ_map = makeUpFullActivMap(activ_map, ksize)

            up_layer = self.model.getUpLayer(layer)
            up_layer_tensor = self.model.getLayerTensor(up_layer)
            feed_dict[up_layer_tensor] = activ_map

            

'''
Fieldmaps Helpers

'''


def stackedFieldmaps(model):
    field_maps = {}
    layer_fieldmaps = layerFieldmaps(model)
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

def layerFieldmaps(model):
    field_maps = []
    for layer in model.layers:
        config = model.getConfig(layer)
        type = config.type
        
        if type == 'conv' or type == 'pool':
            if type == 'conv':
                h, w, _, _ = config.ksize
            else:
                _, h, w, _ = config.ksize
            size = (int(h), int(w))
            strides = tuple(config.strides[1:3])
            padding = config.padding
            padding = _negPaddingFromPadStr(padding, size, strides)
            field_maps.append((layer, padding, size, strides))

    return field_maps
            
def _negPaddingFromPadStr(pad_str, ksize, strides):
    # TODO - transform string padding to corresponding numbers
    pad_str = str(pad_str)
    if 'SAME' in pad_str:
        return tuple((s - k) // 2 for k, s in zip(ksize, strides))
    elif 'VALID' in pad_str:
        return (0, 0)


'''
DeConvNet helpers

'''

# make up activ map to full size of output of layer by zeros
def makeUpFullActivMap(activ_maps, ksize):
    kernel_num = ksize[3]
    

    
'''
ID, Name convertor

'''
    
def unitOfLayer(layer, index):
    return "{}_{}".format(layer, index)

def splitUnitID(unit_id):
    spl = unit_id.split('_')
    layer = "{}_{}".format(spl[0], spl[1])
    unit = spl[2]
    return layer, unit
