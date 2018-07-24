import unittest, os, sys
import numpy as np
from easydict import EasyDict as edict

curr_path = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.join(curr_path, "..")
if root_path not in sys.path:
    sys.path.insert(0, root_path)

from utils.dissection.iou import iou, binarise
from utils.dissection.interp_ref import *
from utils.dissection.upsample import *
from utils.helper.data_loader import BatchLoader
from utils.helper.plotter import maskImage
from utils.helper.file_manager import saveImage
from utils.model.model_agent import ModelAgent
from test_helper import TestBase


class TestInterpRef(TestBase):
    def test_visual_reflect(self):
        bl = BatchLoader(amount=1)
        model = ModelAgent(input_size=1)
        batch = bl.nextBatch()
        imgs = batch[1]
        annos = batch[2]
        activ_maps = model.getActivMaps(imgs, ["conv3_1"])
        field_maps = model.getFieldmaps()
        reflected = reflect(activ_maps, field_maps, annos)
        img = imgs[0]
        saveImage(img, os.path.join(PATH.TEST.ROOT, "raw_img.jpg"))
        for unit, ref in reflected.items():
            ref = np.asarray(ref[0])
            saved = np.zeros(shape=ref.shape+(3,))
            indices = np.argwhere(ref>0)
            saved[indices[:,0], indices[:,1]] = [255, 0, 0]
            saveImage(saved, os.path.join(PATH.TEST.ROOT, unit+".jpg"))

class TestUpsample(TestBase):
    def test_upsampleL(self):
        fieldmap = ((0,0), (1,1), (2,2))
        activ = [
            [0,1],
            [1,0]
        ]
        activ = np.asarray([activ])
        shape = (8, 8)
        input = (4, 4)
        reduc = int(round(input[0] / float(shape[0])))

        upsam = upsampleL(fieldmap, activ)
        print (upsam)

    def test_centered_arange(self):
        field_map = ((0,0), (1,1), (2,2))
        activation_shape = (2, 2)
        reduction = 2

        ay, ax = centered_arange(field_map, activation_shape, reduction)
        print (ay, ax)
        
        
class TestIoU(TestBase):

    def test_iou(self):
        mask1 = np.asarray([[0,0,0],[0,1,0],[0,0,0]])
        mask2 = np.asarray([[0,0,0],[1,1,1],[0,1,0]])
        i = iou(mask1, mask2)
        
        print ("IoU = {}".format(i))
        self.assertEqual(i, 0.25)

    def test_binarise(self):
        mask = np.asarray([[2,2], [-1,0]])
        mask = binarise(mask)

        self.assertEqual(mask, [[1, 1], [0, 0]])
        
if __name__ == "__main__":
    unittest.main()
