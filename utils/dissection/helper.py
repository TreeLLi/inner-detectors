'''
Dissection Helper

To provide auxiliary functions, like IoU calculation and quantile calculation

'''

import numpy as np


'''
IoU

'''

def iou(mask1, mask2, binary=True):
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


'''
Normalisation

'''


def binarise(a, per=0, sequence=False):
    if not sequence:
        _a = a.copy()
        a[_a > per] = 1
        a[_a <= per] = 0
    else:
        for idx, _a in enumerate(a):
            _per = per[idx]
            binarise(_a, _per)


def binariseQuantile(a, per, sequence=False):
    quan = quantile(a, per, sequence)
    binarise(a, quan, sequence)
    

'''
Quantile

'''

def quantile(a, per, sequence=False):
    if not sequence:
        return np.percentile(a, per, interpolation="linear")
    else:
        quans = []
        for _a in a:
            quans.append(np.percentile(_a, per, interpolation="linear"))
        return quans


'''
Filter

'''

def quanFilter(a, per, sequence=False):
    quan = quantile(a, per, sequence)
    _filter(a, quan, sequence)

def _filter(a, quan, sequence=False):
    if not sequence:
        a[a < quan] = 0
    elif isinstance(quan, list):
        for _a, _quan in zip(a, quan):
            _filter(_a, _quan)
    else:
        for _a in a:
            _filter(_a, quan)
