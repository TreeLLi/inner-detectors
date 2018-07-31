'''
Data loader:

Manage the batch loading of data concerning different purposes and datasets

Batch structure:
-TODO:

'''

import os, sys
import numpy as np
import time

from os.path import exists

curr_path = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.join(curr_path, "../..")
if root_path not in sys.path:
    sys.path.insert(0, root_path)

from src.config import PATH, CONFIG, isPASCAL
from utils.helper.data_processor import preprocessImage, preprocessAnnos
from utils.helper.data_mapper import *
from utils.helper.dstruct_helper import splitNumber
from utils.helper.file_manager import loadImage, loadObject, saveObject
from utils.helper.anno_parser import parsePASCALPartAnno


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
    annos = parsePASCALPartAnno(directory, file_name, mappings, mapClassID)
    
    return img, annos
    

'''
Load data as batches

'''

class BatchLoader(object):

    def __init__(self, sources=["PASCAL"], batch_size=10, amount=None, mode=None):
        self.batch_size = batch_size

        self.dataset = loadObject(PATH.DATA.IMG_MAP)
        if amount is not None and len(self.dataset) > amount:
            # discard remaining datasets, since the specified amount is reached
            self.data = self.dataset[:amount]
            self.backup = self.dataset[amount:]
        else:
            self.data = self.dataset.copy()
            self.backup = None
            
        self.amount = len(self.data)
        self.batch_id = 0

        self.mode = mode
        if mode == "classes":
            classes = getClasses()
            if self.amount >= len(classes):
                split = splitNumber(self.amount, len(classes))
                self.cls_counts = {cls : split[i] for i, cls in enumerate(classes)}
            else:
                self.cls_counts = {cls : 1 for cls in classes}
                
    def __bool__(self):
        finished = self.size <= 0
        if finished:
            self.finish()

        return not finished

    @property
    def size(self):
        return len(self.data)
    
    def nextBatch(self, amount=None):
        self.batch_id += 1
        print ("Batch {}: loading...".format(self.batch_id))
        start = time.time()
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
                s_name = sample[0]
                s_source = sample[1]
                s_id = sample[2]
                if self.mode == "classes":
                    # mode 'classes': check if adding this sample
                    # will destroy balance of classes distribution
                    classes = getImageClasses(s_id)
                    remgs = [self.cls_counts[cls]-1 for cls in classes]
                    if any(c < -1 for c in remgs):
                        # sample causing imbalance classes distribution
                        # ignore it and load one more
                        num += 1
                        if self.backup:
                            # if backup data exists, move one to supply
                            self.data.append(self.backup[0])
                            self.backup = self.backup[1:]
                        continue
                
                if isPASCAL(s_source):
                    img, annos = fetchDataFromPASCAL(s_name)
                    
                img = preprocessImage(img)
                annos = preprocessAnnos(annos)
                if len(annos) == 0:
                    # skip the sample without any annotation
                    # increase counter 'num' to load images in next loop
                    num += 1
                else:
                    batch[0].append(s_id)
                    batch[1].append(img)
                    batch[2].append(annos)

                    if DESCRIBE_DATA:
                        describeData(annos, des)
                        
                    if self.mode == "classes":
                        for cls in classes:
                            self.cls_counts[cls] -= 1
                        if all(c <= 0 for c in self.cls_counts.values()):
                            # finished, exist batch loading,
                            # ignore remaining data
                            self.data = []
                            num = 0
                            break
                        
        batch[1] = np.asarray(batch[1])
        self.reportProgress(len(batch[1]), start)
        return batch

    def reportProgress(self, num, start):
        finished = self.batch_id * self.batch_size
        progress = 100 * float(finished) / self.amount
        dur = time.time() - start
        effi = dur / num
        report = ("Batch {}: load {} samples, {}/{}({:.2f}%), taking {:.2f} sec. ({:.2f} sec. / sample)"
                  .format(self.batch_id,
                          num,
                          finished,
                          self.amount,
                          progress,
                          dur,
                          effi))
        print (report)

    def finish(self):
        if not exists(PATH.DATA.CLS_MAP):
            print ("Class map: saved.")
            saveObject(class_map, PATH.DATA.CLS_MAP)
        if not exists(PATH.DATA.IMG_CLS_MAP):
            print ("Image class map: saved")
            saveObject(img_cls_map, PATH.DATA.IMG_CLS_MAP)
        if DESCRIBE_DATA:
            print ("Data statistics: saved")
            saveObject(sortAsClass(des), PATH.DATA.STATISTICS)
