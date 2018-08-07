import scipy.io as sio
import numpy as np
import os, sys

curr_path = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.join(curr_path, "../..")
if root_path not in sys.path:
    sys.path.insert(0, root_path)

from utils.helper.data_mapper import getClassID, mapClassID


'''
structures of annotations:


'''


'''
Parse
'''

def parsePASCALPartAnno(directory, file_name, mappings):
    path = os.path.join(directory, file_name)
    data = sio.loadmat(path)
    data = data['anno']
    file_id = str(data[0][0][0][0])
    
    class_map = mappings[0]
    if mappings[1] is not None:
        img_cls_map = mappings[1]
        img_cls_map.append([file_id])
        _img_cls_map = img_cls_map[-1]
    
    annos = []
    for cls in data[0][0][1][0]:
        cls_name = str(cls[0][0])
        cls_mask = cls[2]

        cls_id = getClassID(cls_name, class_map)
        if cls_id is None:
            class_map.append(cls_name)
            cls_id = len(class_map) - 1
            
        annos.append([cls_id, cls_mask])

        if mappings[1] is not None and cls_id not in _img_cls_map[1:]:
            _img_cls_map.append(cls_id)
            
        try:
            # access the list of part masks
            # if no part masks, cast exception and exit
            part_masks = cls[3][0]
            part_map = class_map[cls_id]
            if len(part_masks)>0 and type(part_map) != list:
                class_map[cls_id] = [class_map[cls_id]]
                part_map = class_map[cls_id]
                
            for part in part_masks:
                part_name = str(part[0][0]).split('_')[0]
                part_mask = part[1]

                if part_name not in part_map:
                    part_map.append(part_name)
                part_id = mapClassID([cls_id, part_map.index(part_name)])
                annos.append([part_id, part_mask])
        except:
            continue
        
    return annos
