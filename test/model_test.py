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
from src.config import PATH

class TestVGG16(unittest.TestCase):
    
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
        with tf.Session() as sess:
            prob = sess.run(vgg16.prob, feed_dict=feed_dict)
            self.assertTrue(prob.any())
            self.assertTrue(vgg16.pool5.shape == (1, 7, 7, 512))


class TestModelAgent(unittest.TestCase):

    def test_init(self):
        model = ModelAgent()
        self.assertTrue(isinstance(model.model, Vgg16))

    def test_get_activ_maps(self):
        bl = BatchLoader(amount=1)
        batch = bl.nextBatch()
        imgs = batch.imgs

        model = ModelAgent()
        activ_maps = model.getActivMaps(imgs)

        # TODO - verify activation maps
            
if __name__ == "__main__":
    unittest.main()
