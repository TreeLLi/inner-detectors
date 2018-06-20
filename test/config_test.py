'''
Configuration test

To test if the configuration loaded well

'''

import unittest, os, sys

curr_path = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.join(curr_path, "..")
if root_path not in sys.path:
    sys.path.insert(0, root_path)

from src.config import *
from test_helper import TestBase
from utils.model.vgg16 import Vgg16

class TestConfig(TestBase):

    def test_isVGG16(self):
        model = "Vgg16"
        self.assertTrue(isVGG16(model))

        model = Vgg16(PATH.MODEL.VGG16.PARAM)
        self.assertTrue(isVGG16(model))

if __name__ == "__main__":
    unittest.main()
