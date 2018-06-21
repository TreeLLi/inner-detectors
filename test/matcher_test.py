'''
Test activations and annotations matcher:

'''

import unittest, os, sys
from easydict import EasyDict as edict
import numpy as np

curr_path = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.join(curr_path, "..")
if root_path not in sys.path:
    sys.path.insert(0, root_path)

from test_helper import TestBase
from src.indr_matcher import *


class TestMatcher(TestBase):

    def test_single_match(self):
        activ = np.asarray([[0,0],[1,0]])
        annos = []
        annos.append(edict({
            "name" : "test",
            "category" : "object",
            "mask" : np.asarray([[0,0],[1,0]])
        }))
        annos.append(edict({
            "name" : "test",
            "category" : "object",
            "mask" : np.asarray([[0,0],[0,0]])
        }))
        annos.append(edict({
            "name" : "part",
            "category" : "part",
            "partof" : "object",
            "mask" : np.asarray([[1,1],[1,1]])
        }))
        
        matches = matchActivAnnos(activ, annos)

        self.assertEqual(matches.test.iou, 0.5)
        self.assertEqual(matches.test.count, 2)
        self.assertEqual(matches.part.iou, 0.25)
        self.assertEqual(matches.part.category, "part")


    def test_multiple_matches(self):
        print ("shit")


    def test_weighted_iou(self):
        iou_1 = 0
        iou_2 = 1
        count_1 = 1
        count_2 = 1

        iou = weightedIoU(iou_1, count_1, iou_2, count_2)
        self.assertEqual(iou, 0.5)
        
if __name__ == "__main__":
    unittest.main()
