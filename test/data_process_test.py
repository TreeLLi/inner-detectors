'''
Data processor test

'''

import unittest, os, sys

curr_path = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.join(curr_path, "..")
if root_path not in sys.path:
    sys.path.insert(0, root_path)

from test_helper import TestBase
from src.config import PATH

from utils.helper.data_processor import *
from utils.helper.dstruct_helper import *
from utils.helper.data_loader import BatchLoader
from utils.helper.data_mapper import getClassNames
from utils.helper.file_manager import saveImage


'''
Test Suits

'''

class TestDataProcessor(TestBase):
    
    def test_preprocess_image(self):
        self.log()
        path = PATH.DATA.PASCAL.IMGS
        image = loadImage(path, "2008_004198.jpg")
        processed = preprocessImage(image)

        self.assertShape(processed, CONFIG.MODEL.INPUT_DIM)

    def test_preprocess_annos(self):
        self.log()
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

    def test_patch(self):
        self.log()
        bl = BatchLoader(amount=10)
        batch = bl.nextBatch()
        imgs = batch[1]
        annos = batch[2]

        imgs, aids = patch(imgs, annos)
        anames = getClassNames(aids)

        path = os.path.join(PATH.TEST.ROOT, "patch")
        idx = 0
        for img, aname in zip(imgs, anames):
            idx += 1
            file_name = "{}_{}.jpg".format(aname, idx)
            file_path = os.path.join(path, file_name)
            saveImage(img, file_path)


class TestDstructHelper(TestBase):
    def test_sort_dict(self):
        self.log()
        dic = {
            2 : 1,
            3 : 2,
            1 : 3,
            4 : 4
        }
        sort = sortDict(dic, by_key=False, descending=True)
        self.assertEqual(sort, [(4,4), (1,3), (3,2), (2,1)])
        
        sort = sortDict(dic, by_key=True, descending=True)
        self.assertEqual(sort, [(4,4), (3,2), (2,1), (1,3)])

    def test_split_dict(self):
        self.log()
        dic = {
            1 : 1,
            2 : 2,
            3 : 3,
            4 : 4
        }

        split = splitDict(dic, 4)
        self.assertLength(split, 4)
        self.assertLength(split[-1], 1)

        split = splitDict(dic, 3)
        self.assertLength(split, 3)
        amount = sum([len(x) for x in split])
        self.assertEqual(amount, 4)

        split = splitDict(dic, base=2)
        self.assertLength(split, 2)
        self.assertEqual(split[0], {1:1, 2:2})

        split = splitDict(dic, base=3)
        self.assertLength(split, 2)
        self.assertEqual(split[-1], {4:4})

    def test_split_list(self):
        self.log()
        lis = [1, 2, 3, 4]
        
        split = splitList(lis, 4)
        self.assertLength(split, 4)
        self.assertLength(split[-1], 1)

        split = splitList(lis, 3)
        self.assertLength(split, 3)
        amount = sum([len(x) for x in split])
        self.assertEqual(amount, 4)

    def test_filter_dict(self):
        self.log()
        dic = {
            1 : {2 : 3, 5 : 5},
            2 : {3 : 4},
            5 : 2
        }
        dic = filterDict(dic, [2])
        self.assertEqual(dic, {1:{2:3}, 2:{3:4}})
        
    def test_reverse_dict(self):
        self.log()
        dic = {
            1 : {
                'a' : 1,
                'b' : 1
            },
            2 : {
                'a' : 2,
                'c' : 2
            }
        }

        rever = {
            'a' : {
                1 : 1,
                2 : 2
            },
            'b' : {1 : 1},
            'c' : {2 : 2}
        }

        returned = reverseDict(dic)
        self.assertEqual(returned, rever)
        
    def test_nest_iter(self):
        self.log()
        dic = {
            1 : {
                1.1 : 1.1,
                1.2 : 1
            },
            2 : {
                2.1 : 2,
                2.2 : 2
            }
        }
        sums = []
        for k_1, k_2, val in nested(dic):
            sums.append(k_1 + k_2 + val)
        self.assertEqual(sums, [3.2, 3.2, 6.1, 6.2])
        
        nest = [
            {1 : [1, 2]},
            {2 : [3, 4]},
            {3 : [5, 6]}
        ]

        results = []
        for it in nested(nest):
            results.append(it)
        self.assertEqual(results[0], [0, 1, 0, 1])
        self.assertEqual(results[-1], [2, 3, 1, 6])

        nest = [
            {1 : [{1:1}, {2:2}]},
            {2 : [{3:3}, {4:4}]},
            {3 : [{3:5}, {6:5}]}
        ]

        results = []
        for it in nested(nest,depth=2):
            results.append(it)
        self.assertEqual(results[0], [0, 1, [{1:1}, {2:2}]])
        self.assertEqual(results[-1], [2, 3, [{3:5}, {6:5}]])

    def test_mean(self):
        self.log()
        dic = {
            1 : {
                1.1 : [2, 1],
                1.2 : [2, 2]
            },
            2 : {
                2.1 : [4, 1]
            },
            3 : {
                3.1 : [0, 0]
            }
        }
        mean = dictMean(dic, key=0)
        self.assertEqual(mean, 2)


'''
Iterator Test

'''
        
nest = {
    1 : {
        1.1 : 1.1,
        1.2 : 1
    },
    2 : {
        2.1 : 2,
        2.2 : 2
    }
}
ni = NestedIterator(nest)
class TestIterator(TestBase):
    def test_get(self):
        val = ni.get([0,0])
        self.assertEqual(val, [1, 1.1, 1.1])

        val = ni.get([1,1])
        self.assertEqual(val, [2, 2.2, 2])

    def test_depth(self):
        depth = ni.depth()
        self.assertEqual(depth, 2)

    def test_increase_indices(self):
        self.assertEqual(ni.indices, [0, 0])

        ni.increaseIndices()
        self.assertEqual(ni.indices, [0, 1])

        ni.increaseIndices()
        self.assertEqual(ni.indices, [1, 0])
        
    def test_pair_iterate(self):
        dic = {1:1, 2:2, 3:3}
        iteration = [set((unit_1, unit_2)) for unit_1, unit_2 in paired(dic)]
        self.assertLength(iteration, 3)
        self.assertContain(iteration, set((1, 2)))
        self.assertContain(iteration, set((1, 3)))
        self.assertContain(iteration, set((2, 3)))
        
if __name__ == "__main__":
    unittest.main()
