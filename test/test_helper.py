'''
Test helper

To provide auxilary functions for verifying and printing convenience
via the form of Base Class

'''

import unittest
import numpy as np

class TestBase(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(TestBase, self).__init__(*args, **kwargs)
        
        self.longMessage = True
        
    def assertListEqual(self, first, second, msg=None):
        first = first.tolist() if isinstance(first, np.ndarray) else first
        second = second.tolist() if isinstance(second, np.ndarray) else second

        return super(TestBase, self).assertListEqual(first, second, msg)

    def assertEmpty(self, expr):
        return self.assertLengthEqual(expr, 0)
    
    def assertNotEmpty(self, expr):
        length = len(expr)
        self.assertGreater(length, 0)

    def assertLengthEqual(self, expr, length):
        self.assertEqual(len(expr), length)

    def assertShapeEqual(self, expr, shape):
        if isinstance(expr, np.ndarray):
            self.assertEqual(expr.shape, shape)
        # TODO - shape comparison of primitive list
