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
        unit = 'conv5'
        batch_matches = edict()
        img_m1 = edict()
        img_m1.leg = edict({
            "iou" : 0.5,
            "count" : 1
        })
        img_m1.arm = edict({
            "iou" : 1,
            "count" : 2
        })
        img_m2 = edict()
        img_m2.leg = edict({
            "iou" : 1,
            "count" : 1
        })
        img_m2.head = edict({
            "iou" : 0,
            "count" : 2
        })
        batch_matches[unit] = [img_m1, img_m2]

        matches = edict()
        matches.conv5 = edict()
        matches.conv5.arm = edict({
            "iou" : 0,
            "count" : 2
        })

        # test None input
        results = combineMatches(None, None)
        self.assertIsNone(results)
        
        combineMatches(matches, batch_matches)

        self.assertEqual(matches.conv5.arm.iou, 0.5)
        self.assertEqual(matches.conv5.arm.count, 4)
        self.assertEqual(matches.conv5.leg.iou, 0.75)
        self.assertEqual(matches.conv5.leg.count, 2)
        self.assertEqual(matches.conv5.head.iou, 0)
        self.assertEqual(matches.conv5.head.count, 2)

    def test_weighted_iou(self):
        iou_1 = 0
        iou_2 = 1
        count_1 = 1
        count_2 = 1

        iou = weightedIoU(iou_1, count_1, iou_2, count_2)
        self.assertEqual(iou, 0.5)

    def test_top_index(self):
        top_n = [('1', 0.3), ('2', 0.2), ('3', 0.1)]
        iou = 0.15
        index = topIndex(top_n, iou)
        self.assertEqual(index, 2)
        self.assertEqual(len(top_n), 3)
        
    def test_matches_filter(self):
        matches = edict()
        conv1 = edict()
        conv1.leg = edict({
            "iou" : 0.5,
            "category" : 'part',
            "count" : 1
        })
        conv1.arm = edict({
            "iou" : 0.2,
            "count" : 2
        })
        conv2 = edict()
        conv2.head = edict({
            "iou" : 0.3,
            "count" : 2
        })
        conv2.leg = edict({
            "iou" : 0,
            "count" : 5
        })
        matches.conv1 = conv1
        matches.conv2 = conv2
        
        filterMatches(matches, top=1, iou_thres=0.3)
        self.assertEqual(len(matches.conv1), 1)
        self.assertEqual(matches.conv1[0].name, "leg")
        self.assertEqual(matches.conv1[0].iou, 0.5)
        self.assertEqual(matches.conv1[0].count, 1)
        self.assertEqual(matches.conv1[0].category, "part")

        self.assertEqual(matches.conv2[0].name, 'head')
        self.assertEqual(matches.conv2[0].iou, 0.3)
        self.assertEqual(matches.conv2[0].count, 2)

    def test_rearrange(self):
        matches = {
            "pool1" : {
                1 : [0.5, 1],
                2 : [0.5, 2]
            },
            "pool2" : {
                2 : [0.2, 1],
                3 : [0.5, 1]
            }
        }
        rearrange = rearrangeMatches(matches)
        self.assertLength(rearrange, 3)
        self.assertEqual(rearrange[2]["pool1"], [0.5, 2])
        self.assertEqual(rearrange[2]["pool2"], [0.2, 1])
        
    def test_text_report(self):
        matches = edict()
        conv1 = edict()
        conv1.leg = edict({
            "iou" : 0.5,
            "category" : 'part',
            "count" : 1
        })
        conv1.arm = edict({
            "iou" : 0.2,
            "count" : 2
        })
        conv2 = edict()
        conv2.head = edict({
            "iou" : 0.3,
            "count" : 2
        })
        conv2.leg = edict({
            "iou" : 0,
            "count" : 5
        })
        matches.conv1 = conv1
        matches.conv2 = conv2
        
        filterMatches(matches, top=2, iou_thres=0.0)
        reportMatchesInText(matches)

    def test_split_activ_maps(self):
        activ_maps = {
            1 : 1,
            2 : 2,
            3 : 3,
            4 : 4
        }

        splited = splitActivMaps(activ_maps, 4)
        self.assertLength(splited, 4)
        self.assertLength(splited[-1], 1)

        splited = splitActivMaps(activ_maps, 3)
        self.assertLength(splited, 3)
        amount = sum([len(x) for x in splited])
        self.assertEqual(amount, 4)
        
if __name__ == "__main__":
    unittest.main()
