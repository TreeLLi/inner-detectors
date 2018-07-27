'''
Data processor test

'''

import unittest, os, sys

curr_path = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.join(curr_path, "..")
if root_path not in sys.path:
    sys.path.insert(0, root_path)

from test_helper import TestBase

from src.config import PATH
from utils.helper.data_processor import *
from utils.helper.data_loader import BatchLoader, getClassNames
from utils.helper.file_manager import saveImage


'''
Test Suits

'''


class TestDataProcessor(TestBase):
    
    def test_preprocess_image(self):
        self.log()
        path = PATH.DATA.PASCAL.IMGS
        image = loadImage(path, "2008_004198.jpg")
        processed = preprocessImage(image)

        self.assertShape(processed, CONFIG.MODEL.INPUT_DIM)

    def test_preprocess_annos(self):
        self.log()
        anno = np.asarray([
            [0,0,1,1],
            [1,1,0,0]
        ])
        anno = [1, anno]
        annos = [anno]
        annos = preprocessAnnos(annos)
        self.assertListEqual(annos[0][1], [[0,1],[1,0]])

        annos[0][1] = np.asarray([
            [0,0,1,1,1,1,1,1,1,1],
            [1,1,0,0,1,1,1,1,1,1]
        ])
        annos = preprocessAnnos(annos)
        self.assertEmpty(annos)

        anno = np.asarray([[0,0],[1,1]])
        annos.append([1, anno])
        annos = preprocessAnnos(annos, 0)
        self.assertLength(annos, 1)

    def test_patch(self):
        self.log()
        bl = BatchLoader(amount=10)
        batch = bl.nextBatch()
        imgs = batch[1]
        annos = batch[2]

        imgs, aids = patch(imgs, annos)
        anames = getClassNames(aids)

        path = os.path.join(PATH.TEST.ROOT, "patch")
        idx = 0
        for img, aname in zip(imgs, anames):
            idx += 1
            file_name = "{}_{}.jpg".format(aname, idx)
            file_path = os.path.join(path, file_name)
            saveImage(img, file_path)

if __name__ == "__main__":
    unittest.main()
