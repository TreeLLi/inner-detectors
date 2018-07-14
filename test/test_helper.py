'''
Test helper

To provide auxilary functions for verifying and printing convenience
via the form of Base Class

'''

import unittest, os, sys
import numpy as np
import inspect

curr_path = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.join(curr_path, "..")
if root_path not in sys.path:
    sys.path.insert(0, root_path)

from src.config import PATH
from utils.helper.file_manager import loadImage

'''
Test Base Class

'''

count = {}

class TestBase(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestBase, self).__init__(*args, **kwargs)
        self.longMessage = True

    '''
    instance getter
    '''
    def getImage(self, img_id="2008_004198.jpg"):
        path = PATH.DATA.PASCAL.IMGS
        image = loadImage(path, img_id)
        return image

    def getImageFile(self):
        return "2008_004198.jpg"

    def getImageId(self):
        return "2008_004198"
    
    '''
    Print

    '''

    def log(self):
        stack = inspect.stack()
        class_name = self.__class__.__name__
        func_name = stack[1][3]
        if class_name not in count:
            count[class_name] = 1
        else:
            count[class_name] += 1
        print ("{} - {}: {}".format(class_name,
                                    count[class_name],
                                    func_name))
        
    '''
    Assertion
    
    '''
        
    def assertListEqual(self, first, second, msg=None):
        first = first.tolist() if isinstance(first, np.ndarray) else first
        second = second.tolist() if isinstance(second, np.ndarray) else second

        return super(TestBase, self).assertListEqual(first, second, msg)

    def assertEmpty(self, expr):
        return self.assertLength(expr, 0)
    
    def assertNotEmpty(self, expr):
        length = len(expr)
        self.assertGreater(length, 0)

    def assertLength(self, expr, length):
        self.assertEqual(len(expr), length)

    def assertShape(self, expr, shape):
        if isinstance(expr, np.ndarray):
            self.assertEqual(expr.shape, shape)
        # TODO - shape comparison of primitive list
