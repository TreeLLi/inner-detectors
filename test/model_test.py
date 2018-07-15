'''
Model test

To test the functionalities of pre-trained models

Currently, without checking correctness, i.e. if the accuray is as research

'''

import unittest, os, sys
import numpy as np
import tensorflow as tf

curr_path = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.join(curr_path, "..")
if root_path not in sys.path:
    sys.path.insert(0, root_path)

from utils.model.vgg16 import Vgg16
from utils.model.model_agent import *
from utils.model.deconvnet import *
from utils.helper.data_loader import BatchLoader
from utils.dissection.upsample import upsampled_shape
from test_helper import TestBase
from src.config import PATH


'''
Test VGG16 model

'''

vgg16 = Vgg16(PATH.MODEL.CONFIG, PATH.MODEL.PARAM, deconv=True)
class TestVGG16(TestBase):
    def test_init(self):
        self.log()
        self.assertIsNotNone(vgg16)
        
    def test_eval(self):
        self.log()
        bl = BatchLoader(amount=10)
        batch = bl.nextBatch()
        imgs = batch[1]

        input = tf.placeholder("float", imgs.shape)
        feed_dict = {input : imgs}
        vgg16.build(input)
        
        with tf.Session() as sess:
            pool5 = sess.run(vgg16.getLayerTensor('pool5'), feed_dict=feed_dict)
            self.assertEqual(pool5.shape, (10, 7, 7, 512))

            
'''
Test ModelAgent 

'''

agent = ModelAgent(input_size=1)
        
class TestModelAgent(TestBase):
    def test_init(self):
        self.log()
        self.assertIsNotNone(agent)

    def test_get_activ_maps(self):
        self.log()
        bl = BatchLoader(amount=1)
        batch = bl.nextBatch()
        imgs = batch[1]
        activ_maps = agent.getActivMaps(imgs, ['pool5'])

        self.assertLength(activ_maps, 512)
        self.assertEqual(activ_maps['pool5_1'].shape, (1, 7, 7))
        self.assertEqual(activ_maps['pool5_2'].shape, (1, 7, 7))

        agent_2 = ModelAgent(input_size=1, deconv=True)
        activ_maps, switches = agent_2.getActivMaps(imgs, ['pool5'])

        self.assertLength(activ_maps, 512)
        self.assertEqual(activ_maps['pool5_1'].shape, (1, 7, 7))
        self.assertLength(switches, 5)
        print (switches['pool5'].shape)

    # def test_layer_unit(self):
    #     self.log()
    #     unit_id = 'pool5_1'
    #     layer = layerOfUnit(unit_id)
    #     self.assertEqual(layer, 'pool5')

    # def test_layer_op(self):
    #     self.log()
    #     op = agent.graph.get_operation_by_name('conv1_1/filter')
    #     layer = layerOfOp(op)
    #     self.assertEqual(layer, 'conv1_1')

    def test_layer_fieldmaps(self):
        self.log()
        field_maps = layerFieldmaps(agent.model)
        _, offset, size, strides = field_maps[0]
        self.assertEqual(size, (3, 3))
        self.assertEqual(offset, (-1, -1))
        self.assertEqual(strides, (1, 1))

        _, offset, size, strides = field_maps[-1]
        self.assertEqual(size, (2, 2))
        self.assertEqual(offset, (0, 0))
        self.assertEqual(strides, (2, 2))

    def test_stacked_fieldmaps(self):
        self.log()
        field_maps = stackedFieldmaps(agent.model)

        out_size = (224, 224)
        field_map = field_maps['conv1_1']
        offset, size, strides = field_map
        self.assertEqual(size, (3, 3))
        self.assertEqual(offset, (-1, -1))
        self.assertEqual(strides, (1, 1))
        
        input_size = upsampled_shape(field_map, out_size)
        self.assertEqual(input_size, (224, 224))

        out_size = (7, 7)
        field_map = field_maps['pool5']
        input_size = upsampled_shape(field_map, out_size)
        self.assertEqual(input_size, (224, 224))

        
        
agent = ModelAgent(input_size=10, deconv=True)
demodel = agent.demodel

class TestDeConvNet(TestBase):
    def test_unpool_layer(self):
        self.log()
        bottom = tf.placeholder(tf.float32, shape=(10, 7, 7, 512))
        name = 'pool5'
        switch = demodel.max_pool_switches[name]
        unpool = demodel.unpoolLayer(bottom, name, switch)
        self.assertIsNotNone(unpool)
        self.assertEqual(unpool.shape.as_list(), [10, 14, 14, 512])
        
    def test_transpose_conv_layer(self):
        self.log()
        bottom = tf.placeholder(tf.float32, shape=(10, 14, 14, 512))
        name = "conv5_3"
        ksize = [3, 3, 512, 512]
        layer = demodel.transposeConvLayer(bottom, name, ksize)
        self.assertIsNotNone(layer)
        self.assertEqual(layer.shape.as_list(), [10, 14, 14, 512])

    def test_build(self):
        demodel.build()
        
if __name__ == "__main__":
    unittest.main()
