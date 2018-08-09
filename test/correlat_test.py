'''
Test correlation analysis program

'''

import unittest, os, sys
import numpy as np

curr_path = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.join(curr_path, "..")
if root_path not in sys.path:
    sys.path.insert(0, root_path)

from test_helper import TestBase
from src.correlation import *


'''
Test Suits

'''

class TestCorrelation(TestBase):

    def test_split(self):
        self.log()
        dic = {
            "conv4_1" : 1,
            "conv5_2" : 2,
            "prob_1" : 1,
            "fc8_1" : 2
        }
        keys = ["prob", "fc8"]
        left, splt = splitDic(dic, keys)

        self.assertEqual(left, {"conv4_1":1, "conv5_2":2})
        self.assertEqual(splt, {"prob_1":1, "fc8_1":2})

    def test_integrate(self):
        self.log()
        dic = {
            "conv4_1" : [1],
            "conv5_2" : [2],
            "prob_1" : [1],
            "fc8_1" : [2]
        }
        _dic = {
            "conv4_1" : [1],
            "conv5_2" : [2],
            "prob_1" : [1],
            "fc8_1" : [2]
        }
        integrate(dic, _dic)

        self.assertEqual(dic, {
            "conv4_1" : [1,1],
            "conv5_2" : [2,2],
            "prob_1" : [1,1],
            "fc8_1" : [2,2]
        })

    def test_splitAttr(self):
        self.log()
        dic = {
            "conv4_1" : np.asarray([[1, 2]]),
            "conv5_2" : np.asarray([[2, 3]]),
            "prob_1" : np.asarray([[1, 4]]),
            "fc8_1" : np.asarray([[2, 6]])
        }
        split = splitAttr(dic)

        self.assertEqual(split[0], {
            "conv4_1" : [[1]],
            "conv5_2" : [[2]],
            "prob_1" : [[1]],
            "fc8_1" : [[2]]
        })
        self.assertEqual(split[1], {
            "conv4_1" : [[2]],
            "conv5_2" : [[3]],
            "prob_1" : [[4]],
            "fc8_1" : [[6]]
        })
        

'''
Main Program

'''

if __name__ == "__main__":
    unittest.main()

                         
