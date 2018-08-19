import inspect
import time

import tensorflow as tf

from utils.model.net import Net

class ConvNet(Net):
    def __init__(self, config_file, param_file, deconv=False):        
        super(ConvNet, self).__init__(config_file, param_file)

        self.deconv = deconv
        if deconv:
            self.max_pool_switches = {}

        
    def build(self, input_size=10, use_cpu=False):
        """
        load variable from npy to build the VGG

        :param rgb: rgb image [batch, height, width, 3] values scaled [0, 1]
        """

        start_time = time.time()
        print("build model started")
        self.loadParams()

        device = self.getDevice(use_cpu)
        with self.graph.as_default():
            with tf.device(device):
                layer = tf.placeholder(tf.float32, shape=(input_size, )+self.input_dim, name="input")
                self.tensors["input"] = layer
                b, g, r = tf.split(axis=3, num_or_size_splits=3, value=layer)
                layer = tf.concat(axis=3, values=[
                    b - self.input_mean[0],
                    g - self.input_mean[1],
                    r - self.input_mean[2],
                ])
                assert layer.get_shape().as_list()[1:] == [224, 224, 3]

                for name in self.layers:
                    config = self.configs[name]
                    layer_type = config.type
                    
                    if layer_type == 'conv':
                        strides = config.strides
                        padding = config.padding
                        layer = self.convLayer(layer, name, strides, padding)
                        self.tensors[name] = layer
                    elif layer_type == 'pool':
                        ksize = config.ksize
                        strides = config.strides
                        padding = config.padding
                        layer = self.maxPoolLayer(layer, name, ksize, strides, padding, self.deconv)
                        self.tensors[name] = layer
                    elif layer_type == 'fc':
                        layer = self.fcLayer(layer, name)
                        if "relu" in config and config.relu:
                            layer = tf.nn.relu(layer)
                            self.tensors[name] = layer
                    elif layer_type == 'classifier':
                        classifier = config.classifier
                        if classifier == 'softmax':
                            layer = tf.nn.softmax(layer, name=name)
                            self.tensors[name] = layer
                    else:
                        print ("Error: unknown layer_type {} for layer {}".format(layer_type, name))
            
        self.params = None
        print(("build model finished: %ds" % (time.time() - start_time)))


    def createFeedDict(self, data):
        input_tensor = self.getInputTensor()
        return {input_tensor : data}

    def maxPoolLayer(self,
                     bottom,
                     name,
                     ksize=[1, 2, 2, 1],
                     strides=[1, 2, 2, 1],
                     padding='SAME',
                     deconv=False):
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
