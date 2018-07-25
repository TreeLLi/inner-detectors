import unittest, os, sys

curr_path = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.join(curr_path, "..")
if root_path not in sys.path:
    sys.path.insert(0, root_path)

from test_helper import TestBase
from src.config import PATH, CONFIG
from utils.helper.anno_parser import parsePASCALPartAnno
from utils.helper.file_manager import *
from utils.helper.data_loader import *

class TestDataLoader(TestBase):

    def test_init(self):
        bl = BatchLoader()
        
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
        self.assertShape(batch[1], (10, 224, 224, 3))
        
    def test_fetch_data_from_pascal(self):
        img_id = self.getImageId()
        img, annos = fetchDataFromPASCAL(img_id)
        
        
    def test_preprocess_image(self):
        path = PATH.DATA.PASCAL.IMGS
        image = loadImage(path, "2008_004198.jpg")
        processed = preprocessImage(image)

        self.assertShape(processed, CONFIG.MODEL.INPUT_DIM)

    def test_preprocess_annos(self):
        anno = np.asarray([
            [0,0,1,1],
            [1,1,0,0]
        ])
        anno = [1, anno]
        annos = [anno]
        annos = preprocessAnnos(annos)
        self.assertListEqual(annos[0][1], [[0,1],[1,0]])

        annos[0][1] = np.asarray([
            [0,0,1,1,1,1,1,1,1,1],
            [1,1,0,0,1,1,1,1,1,1]
        ])
        annos = preprocessAnnos(annos)
        self.assertEmpty(annos)

        anno = np.asarray([[0,0],[1,1]])
        annos.append([1, anno])
        annos = preprocessAnnos(annos, 0)
        self.assertLength(annos, 1)

    def test_get_class_id(self):
        mapping = [None, ["person", "leg"]]
        cls = "leg"
        id = getClassId(cls, mapping)
        self.assertEqual(id, 1001)

    def test_get_class_name(self):
        mapping = [None, None, None, ["person", ["head", "eye"]]]
        cls = "eye"
        id = getClassId(cls, mapping)
        print (id)
        mapped = getClassName(id, mapping)
        self.assertEqual(cls, mapped)

        name = getClassName(id, mapping, True)
        self.assertEqual(name, "person/head/eye")
        
    def test_des_data(self):
        annos = np.asarray([[0,1], [1,0]])
        annos = [[1, annos], [1, annos], [2, annos]]
        annos[1][1] = np.asarray([
            [0, 0, 1, 1],
            [1, 1, 1, 0],
            [2, 3, 1, 1],
            [0, 0, 0, 0]
        ])
        des = describeData(annos, {})
        self.assertLength(des, 2)
        self.assertEqual(des[2], [1, 1, 1, 1, 1, 1, 1, 2])
        self.assertEqual(des[1], [2, 1.5, 2.5, 2, 1.5, 2, 1.625, 5.5])

    def test_finish(self):
        bl = BatchLoader(amount=10)
        batch = bl.nextBatch()
        bl.finish()
        
class TestFileManager(TestBase):

    def test_load_image(self):
        path = PATH.DATA.PASCAL.IMGS
        image = loadImage(path, "2008_004198.jpg")
        self.assertShape(image, (375, 500, 3))
        
    def test_save_load_list_text(self):
        path = os.path.join(PATH.TEST.ROOT, "save_load_list.txt")
        test = [1, 2]
        saveObject(test, path)
        loaded = loadObject(path)
        os.remove(path)
        self.assertEqual(loaded, test)
        
        
    def test_save_load_json(self):
        path = os.path.join(PATH.TEST.ROOT, "save_load_json.json")
        test = {"1" : 2, 2:1}
        saveObject(test, path)
        loaded = loadObject(path)
        os.remove(path)
        self.assertEqual(loaded, test)
        
        
class TestAnnoParser(TestBase):

    def test_parse_pascal_anno(self):
        directory = PATH.DATA.PASCAL.ANNOS
        file_name = "2008_004198.mat"

        mappings = [[], []]
        
        annos = parsePASCALPartAnno(directory, file_name, mappings, mapClassId)
        
if __name__ == "__main__":
    unittest.main()
