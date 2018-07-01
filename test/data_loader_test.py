import unittest, os, sys
from skimage.io import imsave

curr_path = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.join(curr_path, "..")
if root_path not in sys.path:
    sys.path.insert(0, root_path)

from test_helper import TestBase
from src.config import PATH
from utils.helper.anno_parser import parsePASCALPartAnno
from utils.helper.file_manager import loadImage, loadListFromText
from utils.helper.data_loader import *

class TestDataLoader(TestBase):

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
        amount = None
        batch_size = 10
        bl = BatchLoader(batch_size=batch_size, amount=amount)
        self.assertEqual(bl.size, 10103)
        
        batch = bl.nextBatch()
        self.assertNotEmpty(batch)
        self.assertShapeEqual(batch.imgs, (10, 224, 224, 3))
        

    def test_preprocess_image(self):
        path = PATH.DATA.PASCAL.IMGS
        image = loadImage(path, "2008_004198.jpg")
        processed = preprocessImage(image)

        self.assertEqual(processed.shape, (224, 224, 3))
        # out_path = os.path.join(root_path, "2008_004198.jpg")
        # imsave(out_path, processed)

    def test_preprocess_annos(self):
        anno = np.asarray([
            [0,0,1,1],
            [1,1,0,0]
        ])
        anno = edict({
            "mask" : anno
        })
        annos = [anno]
        annos = preprocessAnnos(annos)
        self.assertListEqual(annos[0].mask, [[0,1],[1,0]])

        annos[0].mask = np.asarray([
            [0,0,1,1,1,1,1,1,1,1],
            [1,1,0,0,1,1,1,1,1,1]
        ])
        annos = preprocessAnnos(annos)
        self.assertEmpty(annos)
        
class TestFileManager(TestBase):

    def test_load_image(self):
        path = PATH.DATA.PASCAL.IMGS
        image = loadImage(path, "2008_004198.jpg")
        self.assertShapeEqual(image, (375, 500, 3))
        
    def test_load_list(self):
        path = os.path.join(PATH.ROOT, "src/layers_vgg16.txt")
        layers = loadListFromText(path)
        self.assertNotEmpty(layers)
        
class TestAnnoParser(TestBase):

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
