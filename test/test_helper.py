'''
Test helper

To provide auxilary functions for verifying and printing convenience
via the form of Base Class

'''

import unittest


class TestBase(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(TestBase, self).__init__(*args, **kwargs)
        
        self.longMessage = True
        
