'''
Net

To define the common behaviour of network models

'''


import tensorflow as tf
import numpy as np
from addict import Dict as adict

from utils.helper.file_manager import loadObject

class Net:
    def __init__(self, config_file=None, param_file=None):
        if config_file:
            self.config_file = config_file
            self.loadConfigs()
            
        if param_file:
            self.param_file = param_file
            self.params = None

        self.tensors = {}

    @property
    def graph(self):
        return tf.get_default_graph()
    
    def getConfig(self, layers):
        if isinstance(layers, list):
            return [self.configs[x] for x in layers]
        elif isinstance(layers, str):
            return self.configs[layers]

    def getDevice(self, use_cpu):
        if use_cpu:
            return '/cpu:0'
        else:
            return None
        
    def getTensor(self, layers):
        if isinstance(layers, list):
            return [self.tensors[x] for x in layers]
        elif isinstance(layers, str):
            return self.tensors[layers]

    def getInputTensor(self, layer=None):
        if not layer:
            return self.tensors["input"]
            
        layers = self.layers
        for idx, _layer in enumerate(layers):
            if _layer != layer:
                continue
            # config of previous layer
            pre_layer = layers[idx-1]
            config = self.configs[pre_layer]
            if config.type == "fc":
                return self.tensors["input"]
            else:
                return self.tensors[pre_layer]

    def getProbTensor(self):
        layer = self.layers[-1]
        tensor = self.getTensor(layer)
        return tensor
            
    def createFeedDict(self, data, layer=None, feed_dict=None):
        input_tensor = self.getInputTensor(layer)
        fd = {input_tensor : data}
        if feed_dict:
            fd = {**fd, **feed_dict}
        return fd
        
    def loadConfigs(self):
        configs = loadObject(self.config_file)
        model_configs = configs[0]
        self.input_mean = model_configs["input_mean"]
        self.input_dim = tuple(model_configs["input_dim"])
        self.net_type = model_configs["net_type"]
        
        archi_configs = configs[1:]
        self.layers = []
        self.configs = {}
        for layer_config in archi_configs:
            name = layer_config[0]
            self.layers.append(name)
            self.configs[name] = adict(layer_config[1])
        
    def loadParams(self):
        if not self.params:
            self.params = np.load(self.param_file, encoding='latin1').item()

    def loadConvFilter(self, name):
        if self.net_type == "vgg16":
            return tf.constant(self.params[name][0], name="filter")

    def loadBias(self, name):
        if self.net_type == "vgg16":
            return tf.constant(self.params[name][1], name="biases")

    def loadFCWeight(self, name):
        if self.net_type == "vgg16":
            return tf.constant(self.params[name][0], name="weights")
