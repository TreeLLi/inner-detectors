'''
Data loader:

Manage the batch loading of data concerning different purposes and datasets

Batch structure:
-TODO:

'''

import os, sys
import numpy as np
import time, datetime

from random import shuffle
from os.path import exists
from utils.cocoapi.PythonAPI.pycocotools.coco import COCO as Coco

curr_path = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.join(curr_path, "../..")
if root_path not in sys.path:
    sys.path.insert(0, root_path)

from src.config import *
from utils.helper.data_processor import preprocessImage, preprocessAnnos
from utils.helper.data_mapper import *
from utils.helper.dstruct_helper import splitNumber
from utils.helper.file_manager import loadImage, loadObject, saveObject
from utils.helper.anno_parser import parsePASCALPartAnno
from utils.helper.imagenet_helper import classesOfClassifier


'''
Data Description

'''

if CONFIG.DATA.STATISTICS:
    if not exists(PATH.DATA.STATISTICS):
        des = {}
        DESCRIBE_DATA = True
    else:
        print ("Data statistics have already existed at {}"
               .format(PATH.DATA.STATISTICS))
        DESCRIBE_DATA = False
else:
    DESCRIBE_DATA = False

    
def describeData(annos, des):
    for anno in annos:
        cls_id, cls_mask = anno
        if cls_id in des:
            cls_des = des[cls_id]
        else:
            cls_des = [0 for x in range(8)]
            des[cls_id] = cls_des
        updateClassDes(cls_des, cls_mask)
    return des

def updateClassDes(cls_des, mask):
    row_num = np.asarray([np.sum(row>0) for row in mask])
    row_num = row_num[row_num>0]
    col_num = np.asarray([np.sum(mask[:, i]>0) for i in range(mask.shape[1])])
    col_num = col_num[col_num>0]
    c_1 = cls_des[0]

    ops = [min, max, np.mean]
    vals = [op(num) for num in [row_num, col_num] for op in ops]
    for idx, val in enumerate(cls_des[1:-1]):
        cls_des[idx+1] = weightedVal(val, c_1, vals[idx])
    cls_des[-1] = weightedVal(cls_des[-1], c_1, np.sum(mask>0))
    cls_des[0] += 1

def weightedVal(v_1, c_1, v_2, c_2=1):
    return (v_1*c_1 + v_2*c_2) / (c_1 + c_2)
    

'''
Dataset-specific loading functions

'''

def fetchDataFromPASCAL(img_id):
    postfix = ".jpg"
    directory = PATH.DATA.PASCAL.IMGS
    file_name = img_id + postfix
    img = loadImage(directory, file_name)

    postfix = ".mat"
    file_name = img_id + postfix
    directory = PATH.DATA.PASCAL.ANNOS
    mappings = [class_map, img_cls_map]
    # if img_cls_map exists, then ignore updating in parsing function
    mappings[1] = None if mappings[1] else mappings[1]
    annos = parsePASCALPartAnno(directory, file_name, mappings)
    
    return img, annos

def fetchDataFromCOCO(img_id, subset, coco):
    dir_path = PATH.DATA.COCO.IMGS.format(subset)
    img_id = int(img_id)
    img = coco.loadImgs(img_id)[0]
    img = img['file_name']
    img = loadImage(dir_path, img)

    anno_ids = coco.getAnnIds(imgIds=img_id)
    annos = coco.loadAnns(anno_ids)
    anno_ids = [coco.loadCats(anno['category_id'])[0]['name']
                for anno in annos]
    anno_ids = [getClassID(convert(anno_id)) for anno_id in anno_ids]
    anno_masks = [coco.annToMask(anno) for anno in annos]
    annos = [[_id, _mask] for _id, _mask in zip(anno_ids, anno_masks)]

    return img, annos


'''
Load data as batches

'''

class BatchLoader(object):

    def __init__(self, sources=CONFIG.DIS.DATASETS, batch_size=10, amount=None, random=False, classes=None):
        print ("Data Loader: initialising...")
        self.batch_size = batch_size
        self.batch_id = 0
        self.progress = [0 for x in range(4)]
        
        self.initDatasets(sources, amount)
        
        # special loading mode
        if classes is not None:
            if IMAGENET in sources and type(classes) != list:
                classes = classesOfClassifier()
            self.initModeClasses(classes)
        if random:
            self.randomData()
        
        print ("Data Loader: finish initialisation.")

    def initModeClasses(self, keep):
        if isinstance(keep, list):
            classes = keep
        elif isinstance(keep, int):
            classes = getClasses(order=keep)
            
        if self.amount >= len(classes):
            split = splitNumber(self.amount, len(classes))
            self.cls_counts = {cls : split[i] for i, cls in enumerate(classes)}
        else:
            self.cls_counts = {cls : 1 for cls in classes}

    def randomData(self):
        shuffle(self.data)
        
    def initDatasets(self, sources, amount):
        # config data to be loaded
        self.database = loadObject(PATH.DATA.IMG_MAP)
        self.dataset = []
        # filter samples not belonging to target sources
        for _data in self.database:
            if _data[2] in sources:
                self.dataset.append(_data)
        
        if amount is not None and len(self.dataset) > amount:
            # discard remaining datasets, since the specified amount is reached
            self.data = self.dataset[:amount]
            self.backup = self.dataset[amount:]
        else:
            self.data = self.dataset.copy()
            self.backup = None    
        self.amount = len(self.data)
    
    '''
    Attributes

    '''
    
    def __bool__(self):
        finished = self.size <= 0
        if finished:
            self._finish()

        return not finished

    @property
    def size(self):
        return len(self.data)

    @property
    def img_infos(self):
        if hasattr(self, '_img_infos'):
            return self._img_infos
        else:
            img_infos = loadObject(PATH.DATA.IMG_CLS_MAP)
            self._img_infos = []
            for info in img_infos:
                _info = {}
                _info['id'] = info[0]
                _info['classes'] = info[1:]
                self._img_infos.append(_info)
                
            return self._img_infos
    
    def getCOCO(self, subset):
        # lazy loading to avoid long waiting time for initialisation
        if not hasattr(self, 'cocos'):
            self.cocos = {}
        
        if subset in self.cocos:
            return self.cocos[subset]
        elif subset in ["train", "val"]:
            dir_path = PATH.DATA.COCO.ANNOS
            file_path = dir_path.format(subset)
            self.cocos[subset] = Coco(file_path)
            return self.cocos[subset]
        else:
            raise Exception("Error: invalid subset key for MS COCO")

        
    '''
    Data Loading

    '''
    
    def nextBatch(self, amount=None):
        self.batch_id += 1
        print ("Batch {}: loading...".format(self.batch_id))
        self.progress[0] = time.time()
        # order: [ids, imgs, annos]
        batch = [[], [], []]
        
        num = self.batch_size if amount is None else amount
        while num != 0:
            # adjust the amount of next batch based on the specified amount
            # and the actual remaining number of data
            num = num if self.size > num else self.size
            samples = self.data[:num]
            self.data = self.data[num:]
            num = 0
            
            for sample in samples:
                img_id = sample[1]
                img_source = sample[2]
                img_idx = sample[0]
                if hasattr(self, 'cls_counts'):
                    classes = self.getImageInfo(img_idx)['classes']
                    if all(cls not in self.cls_counts for cls in classes):
                        num += 1
                        continue
                    # mode 'classes': check if adding this sample
                    # will destroy balance of classes distribution
                    remgs = [self.cls_counts[cls]-1 if cls in self.cls_counts else 0
                             for cls in classes]
                    if any(c < -1 for c in remgs):
                        # sample causing imbalance classes distribution
                        # ignore it and load one more
                        num += 1
                        continue
                
                if img_source == PASCAL:
                    img, annos = fetchDataFromPASCAL(img_id)
                elif img_source == COCO:
                    subset = sample[3]
                    coco = self.getCOCO(subset)
                    img, annos = fetchDataFromCOCO(img_id, subset, coco)
                elif img_source == IMAGENET:
                    file_name = "{}.jpg".format(img_id)
                    img = loadImage(PATH.DATA.IMAGENET.IMGS, file_name)
                    annos = None
                    
                img = preprocessImage(img)
                annos = preprocessAnnos(annos)
                if annos is not None and len(annos) == 0:
                    # skip the sample without any annotation
                    # increase counter 'num' to load images in next loop
                    num += 1
                else:
                    batch[0].append(img_idx)
                    batch[1].append(img)
                    if annos:
                        batch[2].append(annos)

                    if DESCRIBE_DATA:
                        describeData(annos, des)
                        
                    if hasattr(self, 'cls_counts'):
                        for cls in classes:
                            if cls in self.cls_counts:
                                self.cls_counts[cls] -= 1
                        if all(c <= 0 for c in self.cls_counts.values()):
                            # finished, exist batch loading,
                            self.finish()
                            num = 0
                            break
            if self.backup:
                # if backup data exists, supply
                self.data += self.backup[:num]
                self.backup = self.backup[num:]
        
        batch[1] = np.asarray(batch[1])
        self.progress[1] = time.time()
        self.progress[2] = len(batch[1])
        return batch


    '''
    Image info getter

    '''

    def getImageInfo(self, img_idx):
        if isinstance(img_idx, list):
            infos = [self.getImageInfo(idx) for idx in img_idx]
            return infos
        elif isinstance(img_idx, int):
            info = self.img_infos[img_idx]
            return info
        else:
            raise Exception("Error: invalid data type for accessing image info.")

        
    '''
    Finish

    '''

    def finish(self):
        self.data = []
    
    def _finish(self):
        if not exists(PATH.DATA.CLS_MAP):
            print ("Class map: saved.")
            saveObject(class_map, PATH.DATA.CLS_MAP)
        if not exists(PATH.DATA.IMG_CLS_MAP):
            print ("Image class map: saved")
            saveObject(img_cls_map, PATH.DATA.IMG_CLS_MAP)
        if DESCRIBE_DATA:
            print ("Data statistics: saved")
            saveObject(sortAsClass(des), PATH.DATA.STATISTICS)

            
    '''
    Progress
    
    '''
         
    def reportProgress(self):
        batch = self.progress[2]
        finished = self.progress[3] + batch
        self.progress[3] = finished
        prop = 100 * float(finished) / self.amount
        left = self.amount - finished

        dur_load = self.progress[1] - self.progress[0]
        dur_batch = time.time() - self.progress[0]

        eff_load = dur_load / batch
        eff_batch = dur_batch / batch

        left_time = left * eff_batch
        end_time = datetime.datetime.now() + datetime.timedelta(seconds=left_time)
        end_time = end_time.strftime("%X %d/%m")
        
        report = "Batch {}: load {} samples, {}/{}({:.2f}%), remaining {} samples.\n"
        report = report.format(self.batch_id, batch, finished, self.amount, prop, left)
        report += "\t loading {:.2f} sec./sample, total {:.2f} sec./sample, end time {}"
        report = report.format(eff_load, eff_batch, end_time)
        
        print (report)
