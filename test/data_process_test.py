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
from utils.helper.data_loader import BatchLoader
from utils.helper.data_mapper import getClassNames
from utils.helper.file_manager import saveImage
from utils.helper.dstruct_helper import *


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


class TestDstructHelper(TestBase):
    def test_sort_dict(self):
        self.log()
        dic = {
            2 : 1,
            3 : 2,
            1 : 3,
            4 : 4
        }
        sort = sortDict(dic, by_key=False, descending=True)
        self.assertEqual(sort, [(4,4), (1,3), (3,2), (2,1)])
        
        sort = sortDict(dic, by_key=True, descending=True)
        self.assertEqual(sort, [(4,4), (3,2), (2,1), (1,3)])

    def test_split_dict(self):
        self.log()
        dic = {
            1 : 1,
            2 : 2,
            3 : 3,
            4 : 4
        }

        splited = splitDict(dic, 4)
        self.assertLength(splited, 4)
        self.assertLength(splited[-1], 1)

        splited = splitDict(dic, 3)
        self.assertLength(splited, 3)
        amount = sum([len(x) for x in splited])
        self.assertEqual(amount, 4)

    def test_split_list(self):
        self.log()
        lis = [1, 2, 3, 4]
        
        split = splitList(lis, 4)
        self.assertLength(split, 4)
        self.assertLength(split[-1], 1)

        split = splitList(lis, 3)
        self.assertLength(split, 3)
        amount = sum([len(x) for x in split])
        self.assertEqual(amount, 4)
        
    def test_reverse_dict(self):
        self.log()
        dic = {
            1 : {
                'a' : 1,
                'b' : 1,
            },
            2 : {
                'a' : 2,
                'c' : 2
            }
        }

        rever = {
            'a' : {
                1 : 1,
                2 : 2
            },
            'b' : {1 : 1},
            'c' : {2 : 2}
        }

        returned = reverseDict(dic)
        self.assertEqual(returned, rever)
        
    def test_nest_iter(self):
        self.log()
        dic = {
            1 : {
                1.1 : 1.1,
                1.2 : 1
            },
            2 : {
                2.1 : 2,
                2.2 : 2
            }
        }
        sums = []
        for k_1, k_2, val in nested(dic):
            sums.append(k_1 + k_2 + val)
            
        self.assertEqual(sums, [3.2, 3.2, 6.1, 6.2])
        
            
if __name__ == "__main__":
    unittest.main()
