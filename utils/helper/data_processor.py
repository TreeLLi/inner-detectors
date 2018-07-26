'''
Data Processor

To pre-process loaded data, including images and annotations

'''


import os, sys
from cv2 import resize
import numpy as np

curr_path = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.join(curr_path, "../..")
if root_path not in sys.path:
    sys.path.insert(0, root_path)

from src.config import CONFIG
from utils.dissection.helper import binarise


'''
Data pre-processing

resize, crop and normalise images but not excluding means of datasets

'''

def preprocessImage(img, target='vgg16'):
    # normalise
    img = cropImage(img)
    img = resize(img, CONFIG.MODEL.INPUT_DIM[:-1])
    
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
            mask = processed[id][1]

    returned = []
    for anno in processed.values():
        # resize it to input size for further comparison
        anno[1] = resize(anno[1], CONFIG.MODEL.INPUT_DIM[:-1])
        binarise(anno[1])
        if np.sum(anno[1]>0) > CONFIG.DIS.ANNO_PIXEL_THRESHOLD:
            returned.append(anno)
        
    return returned

def cropImage(img):
    # crop based on the center
    short_edge = min(img.shape[:2])
    crop_y = int((img.shape[0] - short_edge) / 2)
    crop_x = int((img.shape[1] - short_edge) / 2)
    crop_img = img[crop_y:crop_y+short_edge, crop_x:crop_x+short_edge]
    return crop_img
