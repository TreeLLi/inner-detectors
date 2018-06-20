'''
IoU pixel-wise calculator:

To calculate pixel-wise IoU between two masks

'''

import numpy as np

def iou(mask1, mask2, binary=True):
    if not binary:
        mask1 = binary(mask1)
        mask2 = binary(mask2)

    mask = mask1 + mask2
    count_1 = np.sum(mask == 1)
    count_2 = np.sum(mask == 2)

    return float(count_2) / (count_1 + count_2)

def binary(mask):
    mask[mask>0] = 1
    mask[mask<0] = 0
    
    return mask
