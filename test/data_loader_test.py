import unittest, os, sys

abs_path = os.path.dirname(os.path.abspath(__file__))
par_path = os.path.join(abs_path, "..")
sys.path.append(par_path)

from src.data_loader import BatchLoader, SOURCE, loadBinaryData, loadImages

class TestDataLoader(unittest.TestCase):

    def test_load_binary(self):
        data = loadBinaryData(os.path.join(par_path, "datasets/annos_10.pkl"))
        self.assertTrue(data)

    def test_load_images(self):
        path = os.path.join(par_path, "datasets/VOC2010/JPEGImages")
        images = loadImages(path, ["2008_004198.jpg"])
        self.assertTrue(images)
        
    def test_bool(self):
        bl = BatchLoader(sources=[SOURCE.PASCAL])
        self.assertTrue(bl)

    def test_next(self):
        bl = BatchLoader(sources=[SOURCE.PASCAL])
        next = bl.next()
        self.assertTrue(next)
    
if __name__ == "__main__":
    unittest.main()
