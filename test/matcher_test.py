'''
Test activations and annotations matcher:

'''

import unittest, os, sys
import numpy as np

curr_path = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.join(curr_path, "..")
if root_path not in sys.path:
    sys.path.insert(0, root_path)

from test_helper import TestBase
from src.indr_matcher import *
from src.config import PATH
from utils.dissection.identification import *
from utils.helper.file_manager import loadObject


'''
Test Suits

'''

class TestMatcher(TestBase):

    def test_single_match(self):
        self.log()
        activ = np.asarray([[0,0],[1,0]])
        annos = []
        annos.append([
            1,
            np.asarray([[0,0],[1,0]])
        ])
        annos.append([
            1,
            np.asarray([[0,0],[0,0]])
        ])
        annos.append([
            2,
            np.asarray([[1,1],[1,1]])
        ])
        
        matches = matchActivAnnos(activ, annos)

        self.assertEqual(matches[1][0], 0.5)
        self.assertEqual(matches[1][1], 2)
        self.assertEqual(matches[2][0], 0.25)


    def test_multiple_matches(self):
        self.log()
        unit = 'conv5'
        batch_matches = {}
        img_m1 = {
            "leg" : [0.5, 1],
            "arm" : [1, 2]
        }
        img_m2 = {
            "leg" : [1, 1],
            "head" : [0, 2]
        }
        batch_matches[unit] = [img_m1, img_m2]

        matches = {
            unit : {
                "arm" : [0, 2]
            }
        }

        # test None input
        results = combineMatches(None, None)
        self.assertIsNone(results)
        
        combineMatches(matches, batch_matches)

        self.assertEqual(matches[unit]["arm"][0], 0.5)
        self.assertEqual(matches[unit]["arm"][1], 4)
        self.assertEqual(matches[unit]["leg"][0], 0.75)
        self.assertEqual(matches[unit]["leg"][1], 2)
        self.assertEqual(matches[unit]["head"][0], 0)
        self.assertEqual(matches[unit]["head"][1], 2)

    def test_weighted_iou(self):
        self.log()
        iou_1 = 0
        iou_2 = 1
        count_1 = 1
        count_2 = 1

        iou = weightedIoU(iou_1, count_1, iou_2, count_2)
        self.assertEqual(iou, 0.5)

    def test_top_index(self):
        self.log()
        top_n = [('1', 0.3), ('2', 0.2), ('3', 0.1)]
        iou = 0.15
        index = topIndex(top_n, iou)
        self.assertEqual(index, 2)
        self.assertLength(top_n, 3)
        
    def test_matches_filter(self):
        self.log()
        matches = {
            "conv1" : {
                "leg" : [0.5, 1],
                "arm" : [0.2, 2]
            },
            "conv2" : {
                "head" : [0.3, 2],
                "leg" : [0, 5]
            }
        }
        
        matches = filterMatches(matches, top=1, iou_thres=0.3)
        self.assertLength(matches["conv1"], 1)
        self.assertEqual(matches["conv1"][0][0], "leg")
        self.assertEqual(matches["conv1"][0][1], 0.5)
        self.assertEqual(matches["conv1"][0][2], 1)

        self.assertEqual(matches["conv2"][0][1], 0.3)
        self.assertEqual(matches["conv2"][0][2], 2)

    def test_rearrange(self):
        self.log()
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
        self.log()
        matches = {
            "conv1" : {
                1 : [0.5, 1],
                2 : [0.2, 2]
            },
            "conv2" : {
                3 : [0.3, 2],
                1 : [0, 5]
            }
        }
        
        matches = filterMatches(matches, top=2, iou_thres=0.0)
        reportMatchesInText(matches, PATH.OUT.UNIT_MATCH_REPORT, "unit")
        self.assertExists(PATH.OUT.UNIT_MATCH_REPORT)
        
    def test_split_activ_maps(self):
        self.log()
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

class TestIdent(TestBase):
    def test_organise(self):
        self.log()
        matches = {
            "leg" : {
                "conv5_1_0" : (1, 2),
                "conv5_1_2" : (0, 2),
                "conv5_2_0" : (1, 2)
            },
            "face" : {
                "conv5_1_0" : (1, 2),
                "conv5_1_2" : (1, 2),
                "conv4_2_0" : (1, 2)
            }
        }
        organised = organiseMatches(matches)
        self.assertEqual(organised['leg']['conv5_2'][0], (0, 1, 2))
        self.assertEqual(organised['leg']['conv5_1'][1], (2, 0, 2))

    def test_identification(self):
        self.log()

        ident = loadIdent(sorting=True)
        # matches = loadObject(PATH.OUT.IDE.DATA.UNIT)
        # organised_0 = loadIdent(matches, mode='concept')
        # organised_1 = loadIdent(mode='concept')
        # self.assertEqual(organised_0, organised_1)

        order = 0
        organised = loadIdent(top=10, mode='concept', filtering=order)
        classes = getClasses(order=order)
        self.assertTrue(all(k in classes for k in organised.keys()))
        
    def test_concepts_of_unit(self):
        self.log()
        concepts = conceptsOfUnit("conv5_1_100", top=-10)
        self.assertLength(concepts, 10)
        
if __name__ == "__main__":
    unittest.main()
