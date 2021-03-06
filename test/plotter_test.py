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
from src.config import PATH

from utils.helper.plotter import *
from utils.helper.file_manager import saveImage, saveFigure
from utils.helper.data_loader import BatchLoader
from utils.helper.data_mapper import getClassName


'''
Test Plotter

'''

class TestPlotter(TestBase):
    def test_mask_image(self):
        self.log()
        img = self.getImage("2008_001979.jpg")
        mask = np.zeros(img.shape[:-1])
        
        center = np.asarray(mask.shape) // 2
        size = (100, 100)
        indices = [np.arange(c-s, c+s) for c, s in zip(center, size)]

        for r in indices[0]:
            for c in indices[1]:
                mask[r][c] = 1
                
        masks = [mask]

        mask = np.zeros(img.shape[:-1])
        center = (0, 0)
        size = (200, 200)
        indices = [np.arange(c, c+s) for c, s in zip(center, size)]

        for r in indices[0]:
            for c in indices[1]:
                mask[r][c] = 1
        masks.append(mask)
        
        img = maskImage(img, masks, alpha=0.6)
        # cv2.imshow("test plotter", img)
        # cv2.waitKey(0)
        saveImage(img, os.path.join(root_path, "test_plotter.png"))

    def test_reveal_mask(self):
        self.log()
        bl = BatchLoader(amount=1)
        batch = bl.nextBatch()
        img = batch[1][0]
        annos = batch[2][0]
        for anno in annos:
            aid, mask = anno
            output = revealMask(img, mask)
            path = os.path.join(PATH.TEST.ROOT, "{}.png".format(getClassName(aid, full=True)))
            saveImage(output, path)
        
    def test_plot_figure(self):
        self.log()
        x = [1, 2]
        y = [1, 2]
        ticks = {'x' : ['a', 'b']}
        title = 'test'
        labels = {'x':'x', 'y':'y'}
        plot = plotFigure(x, y, title=title, ticks=ticks, form='spot', show=True, labels=labels)


'''
Main program

'''

if __name__ == "__main__":
    unittest.main()
