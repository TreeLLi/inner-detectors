import unittest, os, sys
import numpy as np
from easydict import EasyDict as edict

curr_path = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.join(curr_path, "..")
if root_path not in sys.path:
    sys.path.insert(0, root_path)

from test_helper import TestBase

from utils.dissection.helper import *
from utils.dissection.interp_ref import *
from utils.dissection.upsample import *


class TestInterpRef(TestBase):
    def test_reflect(self):
        self.log()
    
class TestUpsample(TestBase):
    def test_upsampleL(self):
        self.log()
        fieldmap = ((0, 0), (2, 2), (1, 1))
        activ = [
            [0,1],
            [1,0]
        ]
        activ = np.asarray([activ])
        upsam = upsampleL(fieldmap, activ)
        self.assertShape(upsam, (4, 4, 1))

    def test_centered_arange(self):
        field_map = ((0,0), (1,1), (2,2))
        activation_shape = (2, 2)
        reduction = 2

        ay, ax = centered_arange(field_map, activation_shape, reduction)
        print (ay, ax)
        
        
class TestHelper(TestBase):

    def test_iou(self):
        mask1 = np.asarray([[0,0,0],[0,1,0],[0,0,0]])
        mask2 = np.asarray([[0,0,0],[1,1,1],[0,1,0]])
        i = iou(mask1, mask2)
        
        print ("IoU = {}".format(i))
        self.assertEqual(i, 0.25)

    def test_binarise(self):
        mask = np.asarray([[2,2], [-1,0]])
        binarise(mask)
        self.assertListEqual(mask, [[1, 1], [0, 0]])
        
        mask = np.asarray([[2,4], [-1,0]])
        per = [3, -1]
        binarise(mask, per)
        self.assertListEqual(mask, [[0, 1], [0, 1]])
        
    def test_quantile(self):
        a = [[[1,2,3], [4,5,6]]]
        per = 50
        q = quantile(a, per, sequence=False)
        self.assertEqual(q, 3.5)
        
        q = quantile(a, per, sequence=True)
        self.assertListEqual(q, [3.5])
        
if __name__ == "__main__":
    unittest.main()
