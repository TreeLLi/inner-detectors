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
from utils.helper.file_manager import loadListFromText
from src.config import *


class ModelAgent:

    def __init__(self, model=CONFIG.DIS.MODEL):
        # initialise base model according to the given str
        if isVGG16(model):
            self.model = Vgg16(PATH.MODEL.VGG16.PARAM)

    def getActivMaps(self, imgs):
        if isVGG16(self.model):
            # base model VGG16
            inp = tf.placeholder("float", imgs.shape)
            feed_dict = {inp : imgs}
            self.model.build(inp)

            return _getActivMaps(self.model, feed_dict)



def _getActivMaps(model, data):
    if isinstance(model, Vgg16):
        layers = loadListFromText(PATH.MODEL.VGG16.LAYERS)
        with tf.Session() as sess:
            layers_activ_maps = sess.run([getattr(model, layer) for layer in layers], feed_dict=data)

        activ_maps = edict()
        for idx, layer in enumerate(layers):
            # the dimensions of activation maps: (img_num, width, length, filter_num)
            layer_activ_maps = layers_activ_maps[idx]
            unit_num = layer_activ_maps.shape[3]
            for unit_idx in range(unit_num):
                unit_id = "{}_{}".format(layer, unit_idx)
                activ_maps[unit_id] = layer_activ_maps[:, :, :, unit_idx]

        return activ_maps
