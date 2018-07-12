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
Dataset-specific loading functions

'''

def fetchDataFromPASCAL(identifier):
    postfix = ".jpg"
    directory = PATH.DATA.PASCAL.IMGS
    file_name = identifier + postfix
    img = loadImage(directory, file_name)

    # postfix = ".pkl"
    # directory = PATH.DATA.PASCAL.PRS_ANNOS
    # file_name = identifier + postfix
    # file_path = os.path.join(directory, file_name)
    # if os.path.isfile(file_path):
    #     annos, labels = loadObject(file_path)
    # else:
    postfix = ".mat"
    file_name = identifier + postfix
    directory = PATH.DATA.PASCAL.ANNOS
    annos, labels = parsePASCALPartAnno(directory, file_name)
    # saveObject((annos, labels), file_path)

    return [img, annos, labels]


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

        self.database = File("datasets/pascal_part/pascal_part.h5")
        group_names = ["img", "label", "anno_name", "anno_cate", "anno_mask", "anno_partof"]
        for name in group_names:
            if name not in self.database:
                self.database.create_group(name)
        
    def __bool__(self):
        return self.size != 0

    @property
    def size(self):
        return len(self.data)
    
    def nextBatch(self, amount=None):
        self.batch_id += 1
        print ("Batch {}: loading...".format(self.batch_id))
        start = time.time()
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

                if s_name in self.database:
                    img = self.database.get("img/"+s_name)
                    labels = self.database.get("label/"+s_name)
                    anno_names = self.database.get("anno_name/"+s_name)
                    anno_cates = self.database.get("anno_cate/"+s_name)
                    anno_masks = self.database.get("anno_mask/"+s_name)
                    anno_partof = self.database.get("anno_partof/" + s_name)
                    annos = [{
                        "name" : anno_names[i],
                        "category" : anno_cates[i],
                        "mask" : anno_masks[i],
                        "partof" : anno_partof[i]
                    } for i in range(len(anno_names))]
                    
                else:
                    s_source = sample[1]
                    if s_source == PASCAL:
                        img, annos, labels = fetchDataFromPASCAL(s_name)
                    self.database.get("img").create_dataset(s_name, data=img)
                    self.database.get("label").create_dataset(s_name, data=labels)
                    anno_names = [np.string_(anno['name']) for anno in annos]
                    anno_cates = [np.string_(anno['category']) for anno in annos]
                    anno_masks = [anno['mask'] for anno in annos]
                    anno_partof = [np.string_(anno['partof']) for anno in annos]
                    self.database.get('anno_name').create_dataset(s_name, data=anno_names)
                    self.database.get('anno_cate').create_dataset(s_name, data=anno_cates)
                    self.database.get('anno_mask').create_dataset(s_name, data=anno_masks)
                    self.database.get('anno_partof').create_dataset(s_name, data=anno_partof)
                    
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
        self.reportProgress(len(batch.imgs), start)
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
