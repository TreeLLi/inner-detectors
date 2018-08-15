'''
Test visualisation program

'''

import unittest, os, sys
import numpy as np

curr_path = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.join(curr_path, "..")
if root_path not in sys.path:
    sys.path.insert(0, root_path)

from test_helper import TestBase

from utils.dissection.identification import identification
from src.visualisation import *


'''
Test Suits

'''

class TestVisual(TestBase):
    def test_pool_units(self):
        self.log()
        ident = identification(mode='concept', top=10, organise=True)
        units = poolUnits(ident)
        print (units)

'''
Main program

'''

if __name__ == "__main__":
    unittest.main()
