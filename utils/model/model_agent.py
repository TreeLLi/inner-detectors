'''
Model agent:

To provide a single unified interface to invoke models

'''

import os, sys
import tensorflow as tf
import numpy as np

curr_path = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.join(curr_path, "../..")
if root_path not in sys.path:
    sys.path.append(root_path)

from utils.model.convnet import ConvNet
from utils.model.deconvnet import DeConvNet
from utils.helper.file_manager import loadObject, saveObject
from utils.dissection.upsample import compose_fieldmap, upsampleL
from src.config import *


'''
Model Agent

'''

class ModelAgent:

    def __init__(self,
                 input_size=10,
                 deconv=False):
        self.field_maps = None
        self.input_size = input_size
        self.deconv = deconv
        
        # initialise base model according to the given str
        self.model = ConvNet(PATH.MODEL.CONFIG, PATH.MODEL.PARAM, deconv=deconv)
        self.model.build(input_size)
        
        if self.deconv:
            self.demodel = DeConvNet(self.model)
            self.demodel.build(input_size)

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
        model = self.model
        # generate activation data by forward pass
        if imgs.shape[0] != self.input_size:
            # rebuild model for new input dimension
            self.input_size = imgs.shape[0]
            model.build(self.input_size)

        fetches = {"activs" : {}}
        for layer in layers:
            fetches['activs'][layer] = model.getTensor(layer)
        if self.deconv:
            fetches['switches'] = model.max_pool_switches
        feed_dict = model.createFeedDict(imgs)
        with tf.Session() as sess:
            results = sess.run(fetches, feed_dict=feed_dict)

        activ_maps = {}
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
        switch_feed = {self.demodel.getSwitchTensor(layer) : switch
                     for layer, switch in switches.items()}

        demodel = self.demodel
        for unit_id, activ_map in activ_maps.items():
            layer, unit = splitUnitID(unit_id)

            input_tensor = demodel.getInputTensor(layer)
            kernel_num = input_tensor.shape[-1]
            activ_map = makeUpFullActivMaps(unit, activ_map, kernel_num)
            feed_dict = demodel.createFeedDict(activ_map, layer, switch_feed)

            output_tensor = demodel.output_tensor
            with tf.Session() as sess:
                activ_maps[unit_id] = sess.run(output_tensor, feed_dict=feed_dict)

        return activ_maps


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
def makeUpFullActivMaps(unit, activ_maps, kernel_num):
    shape = activ_maps.shape + (kernel_num, )
    made = np.zeros(shape=shape)
    made[:, :, :, unit] = activ_maps[:, :, :]
    return made

    
'''
ID, Name convertor

'''
    
def unitOfLayer(layer, index):
    return "{}_{}".format(layer, index)

def splitUnitID(unit_id):
    spl = unit_id.split('_')
    if len(spl) == 2:
        return spl[0], int(spl[1])
    elif len(spl) == 3:
        layer = "{}_{}".format(spl[0], spl[1])
        unit = int(spl[2])
        return layer, unit
