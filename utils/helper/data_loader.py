'''
Data loader:

Manage the batch loading of data concerning different purposes and datasets

Batch structure:
-TODO:

'''

from easydict import EasyDict as edict
import os, sys
import numpy as np
from skimage.transform import resize

curr_path = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.join(curr_path, "../..")
if root_path not in sys.path:
    sys.path.insert(0, root_path)

from src.config import PATH, CONFIG, isVGG16
from utils.helper.file_manager import loadImage, loadObject
from utils.helper.anno_parser import parsePASCALPartAnno


# global constants for specify the dataset sources
PASCAL = "PASCAL"
SOURCE = [PASCAL]

'''
Dataset-specific loading functions

'''

def fetchDataFromPASCAL(identifier):
    img_postfix = ".jpg"
    anno_postfix = ".mat"
    img_dir = PATH.DATA.PASCAL.IMGS
    anno_dir = PATH.DATA.PASCAL.ANNOS

    img = loadImage(img_dir, identifier+img_postfix)
    annos, labels = parsePASCALPartAnno(anno_dir, identifier+anno_postfix)

    return img, annos, labels


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
        mask = anno.mask
        name = anno.name
        orig_count = np.sum(mask > 0)
        mask = cropImage(mask)
        crop_count = np.sum(mask > 0)
        retain_ratio = crop_count / float(orig_count)
        if retain_ratio >= mask_thresh:
            # keep the annotations which retain at least
            # 'mask_thresh' ratio of size of masks
            if name not in processed:
                anno.mask = mask
                processed[name] = anno
            else:
                # merge duplicated annos in same images
                print ("Duplicated annos")
                p_mask = processed[name].mask
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
        
        self.data = loadObject(PATH.DATA.MAPS)
        if amount is not None and len(self.data) > amount:
            # discard remaining datasets, since the specified amount is reached
            self.data = self.data[:amount]

        self.amount = amount if amount is not None else len(self.data)
        self.batch_id = 0
        self.sample_id = 0
        
    def __bool__(self):
        return self.size != 0

    @property
    def size(self):
        return len(self.data)
    
    def nextBatch(self, amount=None):
        self.batch_id += 1
        batch = edict({
            "ids" : [],
            "names" : [],
            "imgs" : [],
            "annos" : [],
            "labels" : []
        })
        
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
                    img, annos, labels = fetchDataFromPASCAL(s_name)
                    # add operations for other sources

                img = preprocessImage(img, self.target)
                annos = preprocessAnnos(annos)
                if len(annos) == 0:
                    # skip the sample without any annotation
                    # increase counter 'num' to load images in next loop
                    num += 1
                else:
                    batch.ids.append(self.sample_id)
                    self.sample_id += 1
                    batch.names.append(s_name)
                    batch.imgs.append(img)
                    batch.annos.append(annos)
                    batch.labels.append(labels)

        batch.imgs = np.asarray(batch.imgs)
        self.reportProgress(len(batch.imgs))
        return batch

    def reportProgress(self, num):
        finished = self.amount - self.size
        progress = 100 * float(finished) / self.amount
        report = "Batch {}: load {} samples, {}/{}({:.2f}%)".format(self.batch_id,
                                                                    num,
                                                                    finished,
                                                                    self.amount,
                                                                    progress)
        print (report)
