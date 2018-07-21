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

from utils.model.convnet import ConvNet
from utils.model.model_agent import *
from utils.model.deconvnet import *
from utils.helper.data_loader import BatchLoader
from utils.dissection.upsample import upsampled_shape
from test_helper import TestBase
from src.config import PATH


'''
Test CONVNET model

'''

#convnet = ConvNet(PATH.MODEL.CONFIG, PATH.MODEL.PARAM, deconv=True)
class TestConvNet(TestBase):
    def test_init(self):
        self.log()
        self.assertIsNotNone(convnet)
        
    def test_eval(self):
        self.log()
        bl = BatchLoader(amount=10)
        batch = bl.nextBatch()
        imgs = batch[1]

        convnet.build(10)
        feed_dict = convnet.createFeedDict(imgs)
        
        with tf.Session() as sess:
            pool5 = sess.run(convnet.getTensor('pool5'), feed_dict=feed_dict)
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
        agent.deconv = False
        bl = BatchLoader(amount=1)
        batch = bl.nextBatch()
        imgs = batch[1]
        activ_maps = agent.getActivMaps(imgs, ['pool5'])

        self.assertLength(activ_maps, 512)
        self.assertShape(activ_maps['pool5_1'], (1, 7, 7))
        self.assertShape(activ_maps['pool5_2'], (1, 7, 7))

        agent_2 = ModelAgent(input_size=1, deconv=True)
        activ_maps, switches = agent_2.getActivMaps(imgs, ['pool5'])

        self.assertLength(activ_maps, 512)
        self.assertEqual(activ_maps['pool5_1'].shape, (1, 7, 7))
        self.assertLength(switches, 5)
        print (switches['pool5'].shape)

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

    def test_make_up(self):
        self.log()
        unit = 0
        activ_maps = np.asarray([
            [[1, 2],
             [3, 4],
             ],
            [[5, 6],
             [7, 8]
             ]
        ])
        ksize = 3
        made = makeUpFullActivMaps(unit, activ_maps, ksize)
        self.assertShape(made, (2, 2, 2, 3))
        self.assertEqual(made[0], [[[1,0,0],[2,0,0]],[[3,0,0],[4,0,0]]])

    def test_deconv_maps(self):
        self.log()
        num = 10
        bl = BatchLoader(amount=num)
        batch = bl.nextBatch()
        imgs = batch[1]
        agent = ModelAgent(input_size=num, deconv=True)
        
        probe_layer = ["pool5"]
        activ_maps, switches = agent.getActivMaps(imgs, probe_layer)
        activ_maps = {"pool5_1" : activ_maps["pool5_1"]}
        ref_activ_maps = agent.getDeconvMaps(activ_maps, switches)
        self.assertShape(ref_activ_maps["pool5_1"], [num] + CONFIG.MODEL.INPUT_DIM)
        
        
# agent = ModelAgent(input_size=10, deconv=True)
# demodel = agent.demodel

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
        demodel.loadParams()
        layer = demodel.transposeConvLayer(bottom, name, ksize)
        self.assertIsNotNone(layer)
        self.assertEqual(layer.shape.as_list(), [10, 14, 14, 512])

    def test_build(self):
        self.log()
        demodel.build()

    def test_input_tensor(self):
        self.log()
        demodel.build()
        layer = "pool1"
        tensor = demodel.getInputTensor(layer)
        self.assertEqual(tensor, demodel.tensors["conv2_1"])

        layer = "pool5"
        tensor = demodel.getInputTensor(layer)
        self.assertEqual(tensor, demodel.tensors["input"])
        
if __name__ == "__main__":
    unittest.main()
