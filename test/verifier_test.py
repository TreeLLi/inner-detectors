'''
Verification test

'''

import unittest, os, sys

curr_path = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.join(curr_path, "..")
if root_path not in sys.path:
    sys.path.insert(0, root_path)

from test_helper import TestBase

from src.verifier import *


'''
Test Suits

'''

class TestVerifier(TestBase):

    def test_update_diffs(self):
        attr_diffs = {}
        activ_attrs = {
            "pool5_1" : [[1], [2]],
            "pool5_2" : [[1.5], [3.5]],
        }
        anno_ids = [
            [1, 2],
            [3]
        ]
        updateActivAttrDiffs(attr_diffs, activ_attrs, anno_ids)
        self.assertEqual(attr_diffs["pool5_2"][3][0], [3.5, 0])
        self.assertEqual(attr_diffs["pool5_1"][1][0], [1, 0])
        self.assertEqual(attr_diffs["pool5_1"][2][0], [1, 0])
        self.assertEqual(attr_diffs["pool5_1"][3][0], [2, 0])

        anno_ids = [[3], [2]]
        updateActivAttrDiffs(attr_diffs, activ_attrs, anno_ids, patched=True)
        self.assertEqual(attr_diffs["pool5_1"][1][0], [1, 0])
        self.assertEqual(attr_diffs["pool5_1"][2][0], [1, 2])
        self.assertEqual(attr_diffs["pool5_1"][3][0], [2, 1])
        self.assertEqual(attr_diffs["pool5_2"][3][0], [3.5, 1.5])
        
        
if __name__ == "__main__":
    unittest.main()
