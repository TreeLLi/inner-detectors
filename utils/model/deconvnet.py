'''
DeConvNet

To map activations back to input spaces

'''

import tensorflow as tf
import numpy as np


class DeConvNet:
    
    def __init__(self, model, param_file):
        self.layers = list(reversed(model.layers))
        self.configs = model.configs
        self.max_pool_switches = model.max_pool_switches
        self.tensors = {}
        self.param_file = param_file
        self.params = np.load(self.param_file, encoding='latin1').item()
        self.base_model = model
        
    def build(self, input_size=10):
        if not self.params:
            self.params = np.load(self.param_file, encoding='latin1').item()
            
        layer = tf.placeholder(tf.float32, shape=(input_size, ) + (7, 7, 512), name="input")
        self.tensors["input"] = layer
        for name in self.layers:
            config = self.configs[name]
            layer_type = config.type

            if layer_type == "pool":
                switch = self.max_pool_switches[name]
                strides = config.strides
                layer = self.unpoolLayer(layer, name, switch, strides)
                self.tensors[name] = layer
            elif layer_type == "conv":
                ksize = config.ksize
                strides = config.strides
                padding = config.padding
                layer = self.transposeConvLayer(layer, name, ksize, strides, padding)
                self.tensors[name] = layer
        # last built layer is the output layer
        self.output = layer
        self.params = None

    def getSwitchTensor(self, layer):
        return self.max_pool_switches[layer]

    def getInputTensor(self, layer):
        layers = self.layers
        for idx, lay in enumerate(layers):
            if lay != layer:
                continue
            # config of previous layer
            pre_lay = layers[idx-1]
            config = self.configs[pre_lay]
            if config.type == "fc":
                return self.tensors["input"]
            else:
                return self.tensors[pre_lay]
            
    def unpoolLayer(self, bottom, name, switches, strides=[1, 2, 2, 1]):
        with tf.variable_scope(name):
            b, h, w, c = switches.shape.as_list()
            _, s_h, s_w, _ = strides
            flattened = [tf.scatter_nd(
                tf.reshape(switches[i], [-1, 1]),
                tf.reshape(bottom[i, :, :, :], [-1]),
                [(s_h*h) * (s_w*w) * c],
                name="flatten_{}".format(i)
            )
            for i in range(b)]
            concatenated = tf.concat(flattened, axis=0, name="concat")
            layer = tf.reshape(concatenated,
                               [b, s_h*h, s_w*w, c],
                               name="reconstruction")
        return layer
        
    def transposeConvLayer(self, bottom, name, ksize, strides=[1, 1, 1, 1], padding='SAME'):
        with tf.variable_scope(name):
            layer = tf.nn.relu(bottom, name="relu")
            conv_bias = self.base_model.loadBias(name, self.params)
            layer = tf.nn.bias_add(layer, -conv_bias)
            kernel = self.base_model.loadConvFilter(name, self.params)
            _, _, in_chnl, _ = ksize
            layer = tf.nn.conv2d_transpose(layer,
                                           kernel,
                                           layer.shape.as_list()[:-1] + [in_chnl],
                                           strides,
                                           padding,
                                           name="transpose")
        return layer
