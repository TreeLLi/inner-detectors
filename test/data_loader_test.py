import unittest, os, sys

from itertools import product

curr_path = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.join(curr_path, "..")
if root_path not in sys.path:
    sys.path.insert(0, root_path)

from test_helper import TestBase
from src.config import PATH, CONFIG
from utils.helper.anno_parser import parsePASCALPartAnno
from utils.helper.file_manager import *
from utils.helper.data_loader import *
from utils.helper.data_mapper import *
from utils.helper.imagenet_helper import *

from utils.cocoapi.PythonAPI.pycocotools.coco import COCO


'''
Test Suits

'''

class TestDataLoader(TestBase):
    
    def test_bool(self):
        self.log()
        bl = BatchLoader()
        self.assertTrue(bool(bl))
    
    def test_size(self):
        self.log()
        bl = BatchLoader(amount=10)
        self.assertEqual(bl.size, 10)
        
    def test_next(self):
        self.log()
        bl = BatchLoader()
        batch = bl.nextBatch()
        self.assertShape(batch[1], (10, 224, 224, 3))

        bl = BatchLoader(amount=10, classes=0)
        batch = bl.nextBatch()
        self.assertLength(batch[1], 10)
        # self.assertLength(bl.backup, len(bl.dataset)-12)
        
        bl = BatchLoader(classes=0)
        batch = bl.nextBatch()
        self.assertLength(batch[1], 10)
        self.assertIsNone(bl.backup)

        amount = 100
        bl = BatchLoader(amount=amount, classes=0, random=True)
        self.assertEqual(bl.size, amount)
        batch = bl.nextBatch()
        img_ids = batch[0]
        img_cls_maps = loadObject(PATH.DATA.IMG_CLS_MAP)
        ordered = img_cls_maps[:10]
        ordered = [x[0] for x in ordered]
        self.assertNotEqual(img_ids, ordered)

        bl = BatchLoader(sources=[IMAGENET], amount=10)
        batch = bl.nextBatch()
        self.assertLength(batch[1], 10)

        bl = BatchLoader(sources=[IMAGENET], amount=10, classes=0)
        batch = bl.nextBatch()
        self.assertLength(batch[1], 10)
        
    def test_fetch_data_from_pascal(self):
        self.log()
        img_id = self.getImageId()
        img, annos = fetchDataFromPASCAL(img_id)

    def test_fetch_data_from_coco(self):
        self.log()
        img_id = 139
        subset = "val"
        path = PATH.DATA.COCO.ANNOS.format(subset)
        coco = COCO(path)
        img, annos = fetchDataFromCOCO(img_id, subset, coco)
        self.assertTrue(any(anno[0] in [1,69,73,74,75,76,57,59,61,63]
                            for anno in annos))
        
    def test_des_data(self):
        self.log()
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
        self.log()
        bl = BatchLoader(amount=10)
        batch = bl.nextBatch()
        bl.finish()

        
class TestDataMapper(TestBase):
    
    def test_get_class_id(self):
        self.log()
        mapping = [None, ["person", "leg"]]
        cls = "leg"
        id = getClassID(cls, mapping)
        self.assertEqual(id, 1001)

    def test_get_class_name(self):
        mapping = [None, None, None, ["person", ["head", "eye"]]]
        cls = "eye"
        id = getClassID(cls, mapping)
        print (id)
        mapped = getClassName(id, mapping)
        self.assertEqual(cls, mapped)

        name = getClassName(id, mapping, True)
        self.assertEqual(name, "person/head/eye")

    def test_get_classes(self):
        classes = getClasses(0)
        print ("order 0: ", classes)
        self.assertEqual(classes[0], 1)
        self.assertEqual(classes[19], 20)

        classes = getClasses(1)
        print ("order 1: ", classes)
        self.assertEqual(classes[0], 1001)
        self.assertEqual(classes[1], 2001)

    def test_get_img_classes(self):
        img_id = 0
        classes = getImageClasses(img_id)
        self.assertEqual(classes, [20])

        img_id = "2008_000003"
        classes = getImageClasses(img_id)
        self.assertEqual(classes, [19, 15])
        
        
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
        test = {"1" : 2, "2" : 1}
        saveObject(test, path)
        loaded = loadObject(path)
        os.remove(path)
        self.assertEqual(loaded, test)
        
        
class TestAnnoParser(TestBase):

    def test_parse_pascal_anno(self):
        directory = PATH.DATA.PASCAL.ANNOS
        file_name = "2008_004198.mat"

        mappings = [[], []]
        
        annos = parsePASCALPartAnno(directory, file_name, mappings, mapClassID)


'''
Test ImageNet Helper

'''

class TestImageNetHelper(TestBase):

    def test_wnid_of_name(self):
        self.log()
        name = "Depression, Great Depression"
        wnid = "n15294211"

        self.assertEqual(wnidOfName(name), {wnid})
        self.assertEqual(nameOfWnid(wnid), name)

    def test_super_cates(self):
        self.log()
        wnid = 'n03273913'
        real_sup = ['n04070727', 'n04580493', 'n03528263', 'n02729837', 'n03257877', 'n03093574', 'n03076708', 'n00021939']
        func_sup = superCateIdsOfWnid(wnid)
        self.assertContain(func_sup, real_sup)

        real_sup_names = ['refrigerator', 'white goods', 'home appliance']
        func_names = superCateNamesOfWnid(wnid)
        self.assertContain(func_names, real_sup_names)

    def test_classifier_map(self):
        self.log()
        unmap = set(getClasses())
        mapped = set()
        for idx in range(0, 1000):
            cls = classOfIndice(idx)
            if cls != 0:
                mapped.add(cls)
            if cls in unmap:
                unmap.remove(cls)

        classes = classesOfClassifier()
        mapped = sorted(mapped)
        self.assertEqual(mapped, classes)

    def test_image_urls(self):
        wnid = "n15075141"
        urls = fetchImageUrlsOfWnid(wnid)

        self.assertEqual(urls[-1], "http://farm3.static.flickr.com/2478/3988827219_68c69c9411.jpg")
        
    def test_classes_map(self):
        self.log()
        classes = getClasses()
        classes = [getClassName(cls) for cls in classes]
        wnid_names = loadWnidNames()
        unmapped = set(classes)
        for cls, name in product(classes, wnid_names.values()):
            name = name.split(', ')
            name = [convert(_name) for _name in name]
            if cls in name:
                if cls in unmapped:
                    unmapped.remove(cls)

        print (unmapped)
        
        
if __name__ == "__main__":
    unittest.main()
