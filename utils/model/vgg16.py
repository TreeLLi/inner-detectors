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
        self.data_dict = None
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
        self.layer_tensors = {}
        
    def build(self, rgb):
        """
        load variable from npy to build the VGG

        :param rgb: rgb image [batch, height, width, 3] values scaled [0, 1]
        """

        start_time = time.time()
        print("build model started")
        rgb_scaled = rgb * 255.0

        if not self.data_dict:
            self.data_dict = np.load(self.param_path, encoding='latin1').item()
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

        # self.conv1_1 = self.conv_layer(bgr, "conv1_1")
        # self.conv1_2 = self.conv_layer(self.conv1_1, "conv1_2")
        # self.pool1 = self.max_pool(self.conv1_2, 'pool1', deconv=self.deconv)

        # self.conv2_1 = self.conv_layer(self.pool1, "conv2_1")
        # self.conv2_2 = self.conv_layer(self.conv2_1, "conv2_2")
        # self.pool2 = self.max_pool(self.conv2_2, 'pool2', deconv=self.deconv)

        # self.conv3_1 = self.conv_layer(self.pool2, "conv3_1")
        # self.conv3_2 = self.conv_layer(self.conv3_1, "conv3_2")
        # self.conv3_3 = self.conv_layer(self.conv3_2, "conv3_3")
        # self.pool3 = self.max_pool(self.conv3_3, 'pool3', deconv=self.deconv)

        # self.conv4_1 = self.conv_layer(self.pool3, "conv4_1")
        # self.conv4_2 = self.conv_layer(self.conv4_1, "conv4_2")
        # self.conv4_3 = self.conv_layer(self.conv4_2, "conv4_3")
        # self.pool4 = self.max_pool(self.conv4_3, 'pool4', deconv=self.deconv)

        # self.conv5_1 = self.conv_layer(self.pool4, "conv5_1")
        # self.conv5_2 = self.conv_layer(self.conv5_1, "conv5_2")
        # self.conv5_3 = self.conv_layer(self.conv5_2, "conv5_3")
        # self.pool5 = self.max_pool(self.conv5_3, 'pool5', deconv=self.deconv)

        # self.fc6 = self.fc_layer(self.pool5, "fc6")
        # assert self.fc6.get_shape().as_list()[1:] == [4096]
        # self.relu6 = tf.nn.relu(self.fc6)

        # self.fc7 = self.fc_layer(self.relu6, "fc7")
        # self.relu7 = tf.nn.relu(self.fc7)

        # self.fc8 = self.fc_layer(self.relu7, "fc8")

        # self.prob = tf.nn.softmax(self.fc8, name="prob")

        prev_layer = bgr
        
        for name in self.layers:
            config = self.configs[name]
            layer_type = config.type
            
            if layer_type == 'conv':
                strides = config.strides
                padding = config.padding
                layer = self.conv_layer(prev_layer, name, strides, padding)
            elif layer_type == 'pool':
                ksize = config.ksize
                strides = config.strides
                padding = config.padding
                layer = self.max_pool(prev_layer, name, ksize, strides, padding, self.deconv)
            elif layer_type == 'fc':
                layer = self.fc_layer(prev_layer, name)
                if "relu" in config and config.relu:
                    layer = tf.nn.relu(layer)
            elif layer_type == 'classifier':
                classifier = config.classifier
                if classifier == 'softmax':
                    layer = tf.nn.softmax(prev_layer, name=name)
            else:
                print ("Error: unknown layer_type {} for layer {}".format(layer_type, name))

            self.layer_tensors[name] = layer
            prev_layer = layer
            
        
        self.data_dict = None
        print(("build model finished: %ds" % (time.time() - start_time)))


    def getUpLayer(self, layer):
        idx = self.layers.index(layer)
        if idx < len(self.layers)-1:
            return self.layers[idx+1]
        else:
            raise Exception("Error: try to access an unexisted up layer.")
        
    def getLayerTensor(self, layer):
        if isinstance(layer, list):
            return [self.layer_tensors[x] for x in layer]
        elif isinstance(layer, str):
            return self.layer_tensors[layer]

    def getConfig(self, layer):
        if isinstance(layer, list):
            return [self.configs[x] for x in layer]
        elif isinstance(layer, str):
            return self.configs[layer]

    def getSwitchTensor(self, layer):
        return self.max_pool_switches[layer]
    
    def max_pool(self, bottom, name, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', deconv=False):
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

    def conv_layer(self, bottom, name, strides=[1, 1, 1, 1], padding='SAME'):
        with tf.variable_scope(name):
            filt = self.get_conv_filter(name)

            conv = tf.nn.conv2d(bottom, filt, strides=strides, padding=padding)

            conv_biases = self.get_bias(name)
            bias = tf.nn.bias_add(conv, conv_biases)

            relu = tf.nn.relu(bias)
            return relu

    def fc_layer(self, bottom, name):
        with tf.variable_scope(name):
            shape = bottom.get_shape().as_list()
            dim = 1
            for d in shape[1:]:
                dim *= d
            x = tf.reshape(bottom, [-1, dim])

            weights = self.get_fc_weight(name)
            biases = self.get_bias(name)

            # Fully connected layer. Note that the '+' operation automatically
            # broadcasts the biases.
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

            return fc

    def get_conv_filter(self, name):
        return tf.constant(self.data_dict[name][0], name="filter")

    def get_bias(self, name):
        return tf.constant(self.data_dict[name][1], name="biases")

    def get_fc_weight(self, name):
        return tf.constant(self.data_dict[name][0], name="weights")
