import unittest, os, sys
import numpy as np

curr_path = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.join(curr_path, "..")
if root_path not in sys.path:
    sys.path.insert(0, root_path)

from test_helper import TestBase
from src.config import PATH
    
from utils.dissection.helper import iou, binarise
from utils.dissection.activ_processor import *
from utils.dissection.upsample import *
from utils.helper.data_loader import BatchLoader
from utils.helper.plotter import maskImage
from utils.helper.file_manager import saveImage
from utils.model.model_agent import ModelAgent


'''
Test Suits

'''


class TestActivProcessor(TestBase):
    def test_visual_reflect(self):
        self.log()
        bl = BatchLoader(amount=2)
        model = ModelAgent(input_size=1)
        batch = bl.nextBatch()
        imgs = batch[1][-1:]
        annos = batch[2][-1:]
        activ_maps = model.getActivMaps(imgs, ["conv3_1", "conv4_1", "conv5_1"])
        field_maps = model.getFieldmaps()
        reflected = reflect(activ_maps, field_maps)
        img = imgs[0]
        path = os.path.join(PATH.TEST.ROOT, "output/")
        saveImage(img, os.path.join(path, "raw_img.jpg"))
        for unit, ref in reflected.items():
            ref = np.asarray(ref[0])
            saved = np.zeros(shape=ref.shape+(3,))
            indices = np.argwhere(ref>0)
            saved[indices[:,0], indices[:,1]] = [255, 0, 0]
            saveImage(saved, os.path.join(path, unit+".jpg"))

    def test_activ_attrs(self):
        activ_maps = {
            "pool5_1" : [
                [[2, 2], [2.5, 1.5]],
                [[2, 2], [4, 4]]
            ],
            "pool5_2" : [[0, 0], [-1, 2]]
        }
        attrs = activAttrs(activ_maps)
        self.assertEqual(attrs["pool5_1"], [[2], [3]])
        self.assertEqual(attrs["pool5_2"], [[0.0], [0.5]])

    def test_pearson_coeff(self):
        x = [1, 2, 3]
        y = [1, 2, 3]

        coeff, pvalue = correlation(x, y)
        self.assertEqual(coeff, 1)
        
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
        self.assertEqual(mask, [[1, 1], [0, 0]])
        
        mask = np.asarray([[[2.5, 4.3], [-1.1, 0]]])
        per = [2.2]
        binarise(mask, per, sequence=True)
        self.assertEqual(mask, [[[1, 1], [0, 0]]])
        
    def test_quantile(self):
        a = [[1, 3], [4, 6]]
        per = 25
        q = quantile(a, per, sequence=False)
        self.assertEqual(q, 2.5)

        per = 50
        q = quantile(a, per, sequence=True)
        self.assertEqual(q, [2, 5])
        
if __name__ == "__main__":
    unittest.main()
