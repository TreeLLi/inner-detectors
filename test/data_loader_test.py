import unittest, os, sys

abs_path = os.path.dirname(os.path.abspath(__file__))
par_path = os.path.join(abs_path, "..")
sys.path.append(par_path)

from src.config import PATH
from utils.helper.anno_parser import parsePASCALPartAnno
from utils.helper.file_manager import loadImage
from utils.helper.data_loader import *

class TestDataLoader(unittest.TestCase):

    def test_init(self):
        bl = BatchLoader()
        print (bl.batch_size, bl.amount)
        
        self.assertTrue(bl.batch_size == 10)
        self.assertTrue(bl.amount == 10103)
        self.assertTrue(bl.data)
        
    def test_bool(self):
        bl = BatchLoader(sources=[PASCAL])
        self.assertTrue(bool(bl))
    
    def test_size(self):
        bl = BatchLoader(sources=[PASCAL], amount=10)
        self.assertTrue(bl.size == 10)
        
    def test_next(self):
        bl = BatchLoader()
        batch = bl.nextBatch()
        self.assertTrue(batch)

        self.assertTrue(len(batch.ids) == len(batch.imgs))
        self.assertTrue(len(batch.annos) == 10)
        self.assertTrue(bl.size == 10103-10)

    def test_preprocess(self):
        path = PATH.DATA.PASCAL.IMGS
        image = loadImage(path, "2008_004198.jpg")
        processed = preprocessImage(image)

        print (processed.shape)
        self.assertTrue(processed.shape == (224, 224, 3))

        
class TestFileManager(unittest.TestCase):

    def test_load_image(self):
        path = PATH.DATA.PASCAL.IMGS
        image = loadImage(path, "2008_004198.jpg")
        self.assertTrue(image)
        
        
class TestAnnoParser(unittest.TestCase):

    def test_parse_pascal_anno(self):
        directory = PATH.DATA.PASCAL.ANNOS
        file_name = "2008_004198.mat"
        
        annos, labels = parsePASCALPartAnno(directory, file_name)
        self.assertTrue(len(annos) == 13)
        self.assertTrue(len(labels) == 1)
        self.assertTrue(annos[1].name == "head")
        self.assertTrue(annos[0].category == "object")
        self.assertTrue(annos[2].partof == annos[0].name)

        self.assertTrue(labels[0] == 15)


if __name__ == "__main__":
    unittest.main()
