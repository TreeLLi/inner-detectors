'''
Data loader:

Manage the batch loading of data concerning different purposes and datasets

Batch structure:
-TODO:

'''

from easydict import EasyDict as edict
import os, sys
import numpy as np
import time
from skimage.transform import resize

from h5py import File

curr_path = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.join(curr_path, "../..")
if root_path not in sys.path:
    sys.path.insert(0, root_path)

from src.config import PATH, CONFIG, isVGG16
from utils.helper.file_manager import loadImage, loadObject, saveObject
from utils.helper.anno_parser import parsePASCALPartAnno


# global constants for specify the dataset sources
PASCAL = "PASCAL"
SOURCE = [PASCAL]


'''
Data Mapping

'''

if os.path.isfile(PATH.DATA.CLS_MAP):
    class_map = loadObject(PATH.DATA.CLS_MAP)
else:
    class_map = []

if os.path.isfile(PATH.DATA.IMG_CLS_MAP):
    img_cls_map = loadObject(PATH.DATA.IMG_CLS_MAP)
else:
    img_cls_map = []


def getClassId(cls, mapping=class_map, indices=None):
    try:
        indices.append(mapping.index(cls))
        return mapClassId(indices)
    except:
        if isinstance(mapping, list):
            for idx, m in enumerate(mapping):
                id = getClassId(cls, m, [idx])
                if id is not None:
                    return id
        else:
            return None

def mapClassId(map_indices):
    id = map_indices[0]
    for idx, val in enumerate(map_indices[1:]):
        id += val*1000 if idx == 0 else val*1000*(100**idx)
    return id

def getClassName(cls_id, mapping=class_map):
    id_str = str(cls_id)
    indices = [int(id_str[-3:])]
    id_str = id_str[:-3]
    indices += reversed([int(id_str[i:i+2]) for i in range(0, len(id_str), 2)])
    for idx in indices:
        mapping = mapping[idx]
    return mapping[0] if isinstance(mapping, list) else mapping

def sortAsClass(mapping):
    sorted = []
    for cls in class_map:
        if not cls:
            continue
        
        if isinstance(cls, list):
           for sub_cls in cls:
               cls_id = getClassId(sub_cls)
               if cls_id in mapping:
                   sorted.append([sub_cls] + mapping[cls_id])
        else:
            cls_id = getClassId(cls)
            if cls_id in mapping:
                sorted.append([cls] + mapping[cls_id])
    return sorted


'''
Data Description

'''

if CONFIG.DATA.STATISTICS:
    if not os.path.exists(PATH.DATA.STATISTICS):
        des = {}
        DESCRIBE_DATA = True
    else:
        print ("Data statistics have already existed at {}"
               .format(PATH.DATA.STATISTICS))
        DESCRIBE_DATA = False
else:
    DESCRIBE_DATA = False

    
def describeData(annos):
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
    annos = parsePASCALPartAnno(directory, file_name, mappings, mapClassId)
    
    return img, annos


'''
Data pre-processing

resize, crop and normalise images but not excluding means of datasets

'''

def preprocessImage(img, target='vgg16'):
    # normalise
    img = img / 255.0
    assert (img>=0).all() and (img<=1.0).all()

    img = cropImage(img)
    img = resize(img, CONFIG.MODEL.INPUT_DIM)
    
    return img

# preprocess annos for one single image
def preprocessAnnos(annos, mask_thresh=0.5):
    processed = {}
    for anno in annos:
        mask = anno[1]
        id = anno[0]
        orig_count = np.sum(mask > 0)
        mask = cropImage(mask)
        crop_count = np.sum(mask > 0)
        retain_ratio = crop_count / float(orig_count)
        if retain_ratio >= mask_thresh:
            # keep the annotations which retain at least
            # 'mask_thresh' ratio of size of masks
            if id not in processed:
                anno[1] = mask
                processed[id] = anno
            else:
                # merge duplicated annos in same images
                p_mask = processed[id][1]
                p_mask += mask
                p_mask[p_mask>1] = 1
    return list(processed.values())

def cropImage(img):
    # crop based on the center
    short_edge = min(img.shape[:2])
    crop_y = int((img.shape[0] - short_edge) / 2)
    crop_x = int((img.shape[1] - short_edge) / 2)
    crop_img = img[crop_y:crop_y+short_edge, crop_x:crop_x+short_edge]
    return crop_img
    

'''
Load data as batches

'''

class BatchLoader(object):

    def __init__(self, sources=[PASCAL], target='VGG16', batch_size=10, amount=None):
        self.batch_size = batch_size
        self.target = target
        
        self.data = loadObject(PATH.DATA.IMG_MAP)
        if amount is not None and len(self.data) > amount:
            # discard remaining datasets, since the specified amount is reached
            self.data = self.data[:amount]

        self.amount = amount if amount is not None else len(self.data)
        self.batch_id = 0
        self.sample_id = 0
        
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
                if s_source == PASCAL:
                    img, annos = fetchDataFromPASCAL(s_name)
                    
                img = preprocessImage(img, self.target)
                annos = preprocessAnnos(annos)
                if len(annos) == 0:
                    # skip the sample without any annotation
                    # increase counter 'num' to load images in next loop
                    num += 1
                else:
                    batch[0].append(self.sample_id)
                    self.sample_id += 1
                    batch[1].append(img)
                    batch[2].append(annos)

                    if DESCRIBE_DATA:
                        describeData(annos)

        batch[1] = np.asarray(batch[1])
        self.reportProgress(len(batch[1]), start)
        return batch

    def reportProgress(self, num, start):
        finished = self.amount - self.size
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
        print ("save class_map and img_cls_map")
        saveObject(class_map, PATH.DATA.CLS_MAP)
        saveObject(img_cls_map, PATH.DATA.IMG_CLS_MAP)
        if DESCRIBE_DATA:
            saveObject(sortAsClass(des), PATH.DATA.STATISTICS)
