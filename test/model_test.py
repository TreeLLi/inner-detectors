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
from utils.helper.data_loader import BatchLoader
from test_helper import TestBase
from src.config import PATH

class TestVGG16(TestBase):
    
    def test_init(self):
        try:
            model = Vgg16()
        except:
            print ("Exception: can not find the path for parameters file.")
            model = Vgg16(PATH.MODEL.VGG16.PARAM)

    def test_eval(self):
        bl = BatchLoader(amount=1)
        batch = bl.nextBatch()
        imgs = batch.imgs

        input = tf.placeholder("float", imgs.shape)
        feed_dict = {input : imgs}
        
        vgg16 = Vgg16(PATH.MODEL.VGG16.PARAM)
        vgg16.build(input)
        
        # with tf.Session() as sess:
        #     prob, pool5 = sess.run([vgg16.prob, vgg16.pool5], feed_dict=feed_dict)
        #     self.assertTrue(prob.any())
        #     self.assertEqual(pool5.shape, (1, 7, 7, 512))

        # print (vgg16.conv1_1.name)
        graph = vgg16.conv1_1.graph
        filter = graph.get_operation_by_name('conv5_1/filter')
        conv = graph.get_operation_by_name('conv1_1/Conv2D')
        
        print (filter.outputs[0].shape)
        print (type(filter.node_def), type(filter.op_def))

class TestModelAgent(TestBase):

    def test_init(self):
        model = ModelAgent()
        self.assertIsInstance(model.model, Vgg16)

    def test_get_activ_maps(self):
        bl = BatchLoader(amount=1)
        batch = bl.nextBatch()
        imgs = batch.imgs

        model = ModelAgent()
        activ_maps = model.getActivMaps(imgs)

        # TODO - verify activation maps
        self.assertEqual(len(activ_maps), 1472)
        self.assertEqual(activ_maps.pool1_1.shape, (1, 112, 112))
        self.assertEqual(activ_maps.pool2_1.shape, (1, 56, 56))

    def test_layer_unit(self):
        unit_id = 'pool5_1'
        layer = layerOfUnit(unit_id)
        self.assertEqual(layer, 'pool5')
        
if __name__ == "__main__":
    unittest.main()
