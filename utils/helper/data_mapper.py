'''
Data Mapper

To map data id, classes

'''


import os, sys
from os.path import exists

curr_path = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.join(curr_path, "../..")
if root_path not in sys.path:
    sys.path.insert(0, root_path)

from src.config import PATH
from utils.helper.file_manager import loadObject, saveObject


'''
Global variable

'''

path = PATH.DATA.CLS_MAP
class_map = loadObject(path) if exists(path) else []

path = PATH.DATA.IMG_CLS_MAP
img_cls_map = loadObject(path) if exists(path) else []


'''
Getter

'''

def getClasses(order=0, mapping=class_map, indices=None):
    classes = []
    for idx, cls in enumerate(mapping):
        if idx == 0 or cls is None:
            # skip first element in each order, it is non-sense
            # or belonging to parent order
            continue
        if order != 0:
            if isinstance(cls, list):
                _order = order-1
                _indices = [idx] if indices is None else indices + [idx]
                _classes = getClasses(_order, cls, _indices)
                if _classes:
                    classes += _classes
            else:
                continue
        else:
            _indices = [idx] if indices is None else indices + [idx]
            cls_id = mapClassID(_indices)
            classes.append(cls_id)
    return classes

def getImageClasses(img_id):
    if isinstance(img_id, int):
        return img_cls_map[img_id][1:]
    elif isinstance(img_id, str):
        for img_cls in img_cls_map:
            if img_cls[0] == img_id:
                return img_cls[1:]
        raise Exception("Error: can not find image {}".format(img_id))
    else:
        raise Exception("Error: invalid image identifier.")


'''
Class ID

'''

def getClassID(cls, mapping=class_map, indices=None):
    try:
        indices.append(mapping.index(cls))
        return mapClassID(indices)
    except:
        if isinstance(mapping, list):
            for idx, m in enumerate(mapping):
                _indices = [idx] if indices is None else indices + [idx]
                id = getClassID(cls, m, _indices)
                if id is not None:
                    return id
        else:
            return None

def mapClassID(map_indices):
    id = map_indices[0]
    for idx, val in enumerate(map_indices[1:]):
        id += val*1000 if idx == 0 else val*1000*(100**idx)
    return id


'''
Class Name

'''

def getClassName(cls_id, mapping=class_map, full=False):
    if cls_id < 1000:
        name = mapping[cls_id]
        return name[0] if isinstance(name, list) else name
    
    id_str = str(cls_id)    
    indices = [int(id_str[-3:])]
    id_str = id_str[:-3]
    id_str = "0"+id_str if len(id_str)%2==1 else id_str
    indices += reversed([int(id_str[i:i+2]) for i in range(0, len(id_str), 2)])
    name = ""
    for i, idx in enumerate(indices):
        mapping = mapping[idx]
        if full:
            name += mapping[0] if isinstance(mapping, list) else mapping
            name += "/" if i != len(indices)-1 else ""
        else:
            name = mapping[0] if isinstance(mapping, list) else mapping
    return name

def getClassNames(cls_ids, mapping=class_map, full=False):
    names = []
    cache = {}
    for cls_id in cls_ids:
        cls_id = cls_id[0] if isinstance(cls_id, list) else cls_id
        if cls_id in cache:
            names.append(cache[cls_id])
        name = getClassName(cls_id, mapping, full)
        names.append(name)
        cache[cls_id] = name
    return names


'''
Sorting

'''

def sortAsClass(mapping):
    sorted = []
    for cls in class_map:
        if not cls:
            continue
        
        if isinstance(cls, list):
           for sub_cls in cls:
               cls_id = getClassID(sub_cls)
               if cls_id in mapping:
                   sorted.append([sub_cls] + mapping[cls_id])
        else:
            cls_id = getClassID(cls)
            if cls_id in mapping:
                sorted.append([cls] + mapping[cls_id])
    return sorted


'''
Datasets Label Convertor

'''

CONV = {
    'PASCAL' : {
        "couch" : "sofa",
        "potted plant" : "pottedplant",
        "dining table" : "table",
        "motorcycle" : "motorbike",
        "airplane" : "aeroplane",
        "tv" : "tvmonitor",
        'television' : 'tvmonitor',
        'phone' : 'cell phone',
        'ball' : 'sports ball',
        'wineglass' : 'wine glass',
        'ski' : 'skis'
    },
    'IMAGENET' : {
        'tvmonitor' : 'television',
        'tvmoniter' : 'television',
        'cell phone' : 'phone',
        'sports ball' : 'ball',
        'wine glass' : 'wineglass',
        'skis' : 'ski'
    }
}

def convert(name, dst='PASCAL'):
    if name in CONV:
        return CONV[dst][name]
    else:
        return name
