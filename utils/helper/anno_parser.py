import scipy.io as sio
import numpy as np
import os, pickle

from easydict import EasyDict as edict

'''
structures of annotations:

maps: {file_name : annos}

annos: {class_name : masks}

masks: {'class_id' : id of class,
        'class_mask' : mask of class object,
        'part_masks' : part_masks}

part_masks: [(part_name, part_mask), ...]

'''


'''
Helper functions
'''

def extendListToLength(lis, leng):
    if len(lis) < leng:
        diff = leng - len(lis)
        lis += [None] * diff

'''
Parse
'''

def parsePASCALPartAnno(directory, file_name, mappings, map_cls_id):
    path = os.path.join(directory, file_name)
    data = sio.loadmat(path)
    data = data['anno']
    file_id = data[0][0][0][0]
    
    class_map = mappings[0]
    img_cls_map = mappings[1]
    img_cls_map.append([file_id])
    _img_cls_map = img_cls_map[-1]
    
    annos = []
    for cls in data[0][0][1][0]:
        cls_name = cls[0][0]
        cls_id = cls[1][0][0]
        cls_mask = cls[2]
    
        annos.append([cls_id, cls_mask])

        extendListToLength(class_map, cls_id+1)
        _img_cls_map.append(cls_id)
        if class_map[cls_id] is None:
            class_map[cls_id] = [cls_name]
        
        try:
            # access the list of part masks
            # if no part masks, cast exception and exit
            part_masks = cls[3][0]
            part_map = class_map[cls_id]
            for part in part_masks:
                part_name = part[0][0]
                part_mask = part[1]

                if part_name not in part_map:
                    part_map.append(part_name)
                part_id = map_cls_id([cls_id, part_map.index(part_name)])
                annos.append([part_id, part_mask])
        except:
            continue
        
    return annos
