import inspect
import os

import numpy as np
import tensorflow as tf
import time

from easydict import EasyDict as edict

from utils.helper.file_manager import loadObject

VGG_MEAN = [103.939, 116.779, 123.68]


class Vgg16:
    def __init__(self, config, param_path, deconv=False):        
        self.params = None
        self.param_path = param_path
        self.deconv = deconv

        config = loadObject(config)
        self.layers = []
        self.configs = {}
        for layer_config in config:
            name = layer_config[0]
            self.layers.append(name)
            self.configs[name] = edict(layer_config[1])
        
        self.max_pool_switches = {}
        self.tensors = {}
        
    def build(self, rgb):
        """
        load variable from npy to build the VGG

        :param rgb: rgb image [batch, height, width, 3] values scaled [0, 1]
        """

        start_time = time.time()
        print("build model started")
        rgb_scaled = rgb * 255.0

        if not self.params:
            self.params = np.load(self.param_path, encoding='latin1').item()
            print("npy file loaded")
        
        # Convert RGB to BGR
        red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=rgb_scaled)
        assert red.get_shape().as_list()[1:] == [224, 224, 1]
        assert green.get_shape().as_list()[1:] == [224, 224, 1]
        assert blue.get_shape().as_list()[1:] == [224, 224, 1]
        bgr = tf.concat(axis=3, values=[
            blue - VGG_MEAN[0],
            green - VGG_MEAN[1],
            red - VGG_MEAN[2],
        ])
        assert bgr.get_shape().as_list()[1:] == [224, 224, 3]

        layer = bgr        
        for name in self.layers:
            config = self.configs[name]
            layer_type = config.type
            
            if layer_type == 'conv':
                strides = config.strides
                padding = config.padding
                layer = self.convLayer(layer, name, strides, padding)
            elif layer_type == 'pool':
                ksize = config.ksize
                strides = config.strides
                padding = config.padding
                layer = self.maxPoolLayer(layer, name, ksize, strides, padding, self.deconv)
            elif layer_type == 'fc':
                layer = self.fcLayer(layer, name)
                if "relu" in config and config.relu:
                    layer = tf.nn.relu(layer)
            elif layer_type == 'classifier':
                classifier = config.classifier
                if classifier == 'softmax':
                    layer = tf.nn.softmax(layer, name=name)
            else:
                print ("Error: unknown layer_type {} for layer {}".format(layer_type, name))

            self.tensors[name] = layer
            
        self.params = None
        print(("build model finished: %ds" % (time.time() - start_time)))


    def getUpLayer(self, layer):
        idx = self.layers.index(layer)
        if idx < len(self.layers)-1:
            return self.layers[idx+1]
        else:
            raise Exception("Error: try to access an unexisted up layer.")
        
    def getLayerTensor(self, layer):
        if isinstance(layer, list):
            return [self.tensors[x] for x in layer]
        elif isinstance(layer, str):
            return self.tensors[layer]

    def getConfig(self, layer):
        if isinstance(layer, list):
            return [self.configs[x] for x in layer]
        elif isinstance(layer, str):
            return self.configs[layer]

    def getSwitchTensor(self, layer):
        return self.max_pool_switches[layer]
    
    def maxPoolLayer(self, bottom, name, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', deconv=False):
        if not deconv:
            return tf.nn.max_pool(bottom,
                                  ksize=ksize,
                                  strides=strides,
                                  padding=padding,
                                  name=name)
        else:
            pool, switches = tf.nn.max_pool_with_argmax(bottom,
                                                        ksize=ksize,
                                                        strides=strides,
                                                        padding=padding,
                                                        name=name)
            self.max_pool_switches[name] = switches
            return pool

    def convLayer(self, bottom, name, strides=[1, 1, 1, 1], padding='SAME'):
        with tf.variable_scope(name):
            filt = self.loadConvFilter(name)

            conv = tf.nn.conv2d(bottom, filt, strides=strides, padding=padding)

            conv_biases = self.loadBias(name)
            bias = tf.nn.bias_add(conv, conv_biases)

            relu = tf.nn.relu(bias)
            return relu

    def fcLayer(self, bottom, name):
        with tf.variable_scope(name):
            shape = bottom.get_shape().as_list()
            dim = 1
            for d in shape[1:]:
                dim *= d
            x = tf.reshape(bottom, [-1, dim])

            weights = self.loadFCWeight(name)
            biases = self.loadBias(name)

            # Fully connected layer. Note that the '+' operation automatically
            # broadcasts the biases.
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

            return fc

    def loadConvFilter(self, name, params=None):
        if not params:
            params = self.params
        return tf.constant(params[name][0], name="filter")

    def loadBias(self, name, params=None):
        if not params:
            params = self.params
        return tf.constant(params[name][1], name="biases")

    def loadFCWeight(self, name, params=None):
        if not params:
            params = self.params
        return tf.constant(params[name][0], name="weights")
