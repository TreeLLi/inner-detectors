import unittest, os, sys
import numpy as np

abs_path = os.path.dirname(os.path.abspath(__file__))
par_path = os.path.join(abs_path, "..")
sys.path.append(par_path)

from utils.dissection.iou import iou, binary


class TestIoU(unittest.TestCase):

    def test_iou(self):
        mask1 = np.asarray([[0,0,0],[0,1,0],[0,0,0]])
        mask2 = np.asarray([[0,0,0],[1,1,1],[0,1,0]])
        i = iou(mask1, mask2)
        
        print ("IoU = {}".format(i))
        self.assertTrue(i == 0.25)

    def test_binary(self):
        mask = np.asarray([[2,2], [-1,0]])
        mask = binary(mask)

        print (mask)
        
        self.assertTrue(mask[0][0] == 1)
        
if __name__ == "__main__":
    unittest.main()
