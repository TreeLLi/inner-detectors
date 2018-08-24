'''
ImageNet API Helper

To provide auxiliary functions of ImageNet

'''

import requests
import os, sys

curr_path = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.join(curr_path, "../..")
if root_path not in sys.path:
    sys.path.insert(0, root_path)

from src.config import PATH, URL
from utils.helper.data_mapper import getClasses, getClassNames, convert
from utils.helper.file_manager import loadObject


'''
Conversion between datasets

'''

def classOfIndice(indice):
    if isinstance(indice, list):
        return [classOfIndice(idx) for idx in indice]
    
    mapping = loadClassifierClasses()
    name = mapping[indice]
    wnid = wnidOfName(name)
    sups = superCateNamesOfWnid(wnid)
    cls_ids = getClasses()
    cls_names = getClassNames(cls_ids)
    for idx, cls in enumerate(cls_names):
        if cls in sups:
            return cls_ids[idx]
    return 0

def classesOfClassifier(label=False):
    csf_labels = loadClassifierClasses()
    indices = [x for x in range(len(csf_labels))]
    classes = classOfIndice(indices)
    if label:
        labels = {}
        for idx, cls in enumerate(classes):
            if cls == 0:
                continue
            if cls not in labels:
                labels[cls] = []
            labels[cls].append((idx, csf_labels[idx]))
    classes = set(classes)
    classes.remove(0)
    if label:
        return sorted(classes), labels
    else:
        return sorted(classes)
    

'''
Wnid & Class Name

'''

WNID_NAME = {}
def loadWnidNames():
    global WNID_NAME
    if WNID_NAME:
        return WNID_NAME
    path = PATH.DATA.IMAGENET.WORDS
    wnid_names = loadObject(path, split=False)
    for entry in wnid_names:
        entry = entry.split('\t')
        WNID_NAME[entry[0]] = entry[1]
    return WNID_NAME

ISA_MAP = {}
def loadIsAMap():
    global ISA_MAP
    if ISA_MAP:
        return ISA_MAP
    path = PATH.DATA.IMAGENET.ISA
    isa_map = loadObject(path, split=False)
    for entry in isa_map:
        entry = entry.split(' ')
        sup, sub = entry
        ISA_MAP[sub] = sup
    return ISA_MAP

CSF_CLASS = None
def loadClassifierClasses():
    global CSF_CLASS
    if CSF_CLASS:
        return CSF_CLASS
    path = PATH.MODEL.CLASSES
    mapping = loadObject(path, split=False)
    CSF_CLASS = mapping
    return mapping

def wnidOfName(name):
    wnid_names = loadWnidNames()
    for wnid, _name in wnid_names.items():
        if _name == name:
            return wnid
    raise Exception("Error: name {} to be searched for wnid is not found.".format(name))
    return None

def nameOfWnid(wnid, split=False, conver=True):
    wnid_names = loadWnidNames()
    if wnid in wnid_names:
        name = wnid_names[wnid]
        name = name.split(', ') if split else name
        if conver:
            name = [convert(_name) for _name in name] if split else convert(name)
        return name
    else:
        raise Exception("Error: wnid to be searched is not found.")
    
def superCateIdsOfWnid(wnid):
    isa_map = loadIsAMap()
    if wnid in isa_map:
        sup = isa_map[wnid]
        sup_sup = superCateIdsOfWnid(sup)
        if sup_sup:
            return [sup] + sup_sup
        else:
            return [sup]
    else:
        return None

def superCateNamesOfWnid(wnid):
    sup_wnids = superCateIdsOfWnid(wnid)
    names = []
    for sup_wnid in sup_wnids:
        names += nameOfWnid(sup_wnid, split=True)
    return names


'''
ImageNet API

'''

def fetchImageUrlsOfWnid(wnid):
    url = URL.IMAGENET.IMG_URLS
    params = {"wnid" : wnid}
    results = requests.get(url, params=params)
    urls = results.text
    urls = urls.split('\r')
    urls = [url.strip('\n') for url in urls]
    urls = urls[:-1] # ignore the last empty one caused by '\n'
    return urls

    
