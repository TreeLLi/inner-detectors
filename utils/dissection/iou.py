'''
IoU pixel-wise calculator:

To calculate pixel-wise IoU between two masks

'''

import numpy as np

def iou(mask1, mask2, binary=False):
    if not binary:
        mask1 = binarise(mask1)
        mask2 = binarise(mask2)

    mask = mask1 + mask2
    count_1 = np.sum(mask == 1)
    count_2 = np.sum(mask == 2)

    if count_1 > 0 or count_2 > 0:
        return float(count_2) / (count_1 + count_2)
    else:
        return 0
    
def binarise(mask):
    mask[mask>0] = 1
    mask[mask<0] = 0
    
    return mask
