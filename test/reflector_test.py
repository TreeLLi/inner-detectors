import unittest, os, sys
import numpy as np
from easydict import EasyDict as edict

curr_path = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.join(curr_path, "..")
if root_path not in sys.path:
    sys.path.insert(0, root_path)

from utils.dissection.iou import iou, binary
from utils.dissection.interp_ref import *
from utils.helper.data_loader import BatchLoader
from utils.model.model_agent import ModelAgent
from test_helper import TestBase


bl = BatchLoader(amount=1)
batch = bl.nextBatch()
imgs = batch.imgs
model_agent = ModelAgent()
activ_maps = model_agent.getActivMaps(imgs)


class TestLinearRef(TestBase):

    def test_reflect(self):
        ref_activ_maps = reflect(activ_maps, model_agent)
        
        target_dim = (activ_maps.pool1_1.shape[0],) + (224, 224)
        self.assertEqual(ref_activ_maps.pool1_1.shape, target_dim) 
        
    

class TestIoU(TestBase):

    def test_iou(self):
        mask1 = np.asarray([[0,0,0],[0,1,0],[0,0,0]])
        mask2 = np.asarray([[0,0,0],[1,1,1],[0,1,0]])
        i = iou(mask1, mask2)
        
        print ("IoU = {}".format(i))
        self.assertEqual(i, 0.25)

    def test_binary(self):
        mask = np.asarray([[2,2], [-1,0]])
        mask = binary(mask)

        self.assertEqual(mask, [[1, 1], [0, 0]])
        
if __name__ == "__main__":
    unittest.main()
