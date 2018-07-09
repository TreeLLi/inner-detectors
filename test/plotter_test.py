'''
Image Plotter Test

'''

import unittest, os, sys
import numpy as np

curr_path = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.join(curr_path, "..")
if root_path not in sys.path:
    sys.path.insert(0, root_path)

from test_helper import TestBase
from utils.helper.plotter import *
from utils.helper.file_manager import saveImage

'''
Test Plotter

'''

class TestPlotter(TestBase):
    def test_mask_image(self):
        img = self.getImage("2008_001979.jpg")
        mask = np.zeros(img.shape[:-1])
        
        center = np.asarray(mask.shape) // 2
        size = (100, 100)
        indices = [np.arange(c-s, c+s) for c, s in zip(center, size)]

        for r in indices[0]:
            for c in indices[1]:
                mask[r][c] = 1
                
        masks = [mask]

        img = maskImage(img, masks, alpha=0.1)
        # cv2.imshow("test plotter", img)
        # cv2.waitKey(0)
        saveImage(os.path.join(root_path, "test_plotter.jpg"), img)


'''
Main program

'''

if __name__ == "__main__":
    unittest.main()
