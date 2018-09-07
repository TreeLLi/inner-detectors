'''
Data preparation

To download, organise and generate intermediate description files

'''


import sys
import numpy as np
import time

from os.path import exists, join, dirname, abspath
from random import shuffle
from skimage.io import imread

curr_path = dirname(abspath(__file__))
root_path = join(curr_path, "..")
if root_path not in sys.path:
    sys.path.insert(0, root_path)

from src.config import *
from utils.helper.file_manager import loadObject, saveObject, saveImage, getFilesInDirectory
from utils.helper.anno_parser import parsePASCALPartAnno
from utils.helper.data_mapper import mapClassID, getClassID, getClassName, convert
from utils.helper.dstruct_helper import nested


'''
Download

'''

def downloadDatasets(sources):
    print ("download datasets of sources.")

def downloadImageNet():
    print ("downloading ImageNet dataset.")
    
    classes, labels = classesOfClassifier(True)
    cls_urls = {}
    for cls, _, label in nested(labels, depth=2):
        csf_idx, name = label
        wnid = wnidOfName(name)
        urls = fetchImageUrlsOfWnid(wnid)
        if cls not in cls_urls:
            cls_urls[cls] = []
        cls_urls[cls] += [(url, csf_idx) for url in urls]

    counter = {cls : 0 for cls in classes}
    for cls, data in cls_urls.items():
        shuffle(data)
        _data = data[:100]
        data = data[100:]
        
        for url, csf_idx in _data:
            try:
                img = imread(url)
            except:
                _data.append(data[0])
                data = data[1:]
                continue
            if img is None:
                continue
            idx = counter[cls]
            counter[cls] += 1
            file_name = "{}_{}_{}.jpg".format(getClassName(cls), csf_idx, idx)
            file_path = join(PATH.DATA.IMAGENET.IMGS, file_name)
            saveImage(img, file_path, plugin='skimage')
    

'''
Images ID, Classes Mapping

'''

def mapDatasets(sources):
    maps = [[] for x in range(3)]
    maps[1].append("background")
    sources = sorted(sources)
    for source in sources:
        if source == PASCAL:
            mapPASCAL(maps, source)
        elif source == COCO:
            mapCOCO(maps, source)
            
    maps[0] = [(idx, ) + _map for idx, _map in enumerate(maps[0])]
    paths = [PATH.DATA.IMG_MAP, PATH.DATA.CLS_MAP, PATH.DATA.IMG_CLS_MAP]
    for _map, _path in zip(maps, paths):
        saveObject(_map, _path)
    return maps

def mapPASCAL(maps, source):
    print ("Mapping PASCAL Part dataset...")
    img_ids, cls_map, img_cls_map = maps

    # image id
    anno_dir = PATH.DATA.PASCAL.ANNOS
    data = getFilesInDirectory(anno_dir, "mat")
    data = [(x[x.rfind('/')+1:-4], source) for x in data]
    img_ids += data
    
    # class map
    for img_id in data:
        file_name = img_id[0] + ".mat"
        parsePASCALPartAnno(anno_dir,
                            file_name,
                            [cls_map, img_cls_map])
    print ("Finish mapping PASCAL Part dataset.")
        
def mapCOCO(maps, source):
    print ("Mapping MS-COCO dataset...")
    img_ids, cls_map, img_cls_map = maps
    
    for subset in ["val", "train"]:
        file_path = PATH.DATA.COCO.ANNOS.format(subset)
        coco = loadObject(file_path)

        # class map
        classes = coco['categories']
        _cls_map = {}
        for idx, cls in enumerate(classes):
            name = convert(cls['name'])
            _cls_id = cls['id']
            cls_id = getClassID(name, cls_map)
            if cls_id is None:
                # new class not exists in class_map
                cls_map.append(name)
                _cls_map[_cls_id] = len(cls_map) - 1
            else:
                _cls_map[_cls_id] = cls_id
                
        annos = {}
        for anno in coco['annotations']:
            img_id = anno['image_id']
            cls = _cls_map[anno['category_id']]
            if img_id not in annos:
                annos[img_id] = set()
            annos[img_id].add(cls)

        data = sorted(list(annos.keys()))
        # only count the image_id appearing in the annotations
        # not all image_ids in coco['images']
        for img_id in data:
            img_cls = [img_id] + list(annos[img_id])
            img_cls_map.append(img_cls)
            img_ids.append((img_id, source, subset))

def mapImageNet(maps=None):
    if maps:
        img_ids, _, img_cls_map = maps
    else:
        img_ids = loadObject(PATH.DATA.IMG_MAP)
        img_cls_map = loadObject(PATH.DATA.IMG_CLS_MAP)
    
    img_dir = PATH.DATA.IMAGENET.IMGS
    data = getFilesInDirectory(img_dir, "jpg")
    data = [(x[x.rfind('/')+1:-4], IMAGENET) for x in data]

    idx = len(img_ids)
    for _data in data:
        img_ids.append((idx, ) + _data)
        img_id = _data[0]
        cls = getClassID(img_id.split('_')[0])
        img_cls_map.append([img_id, cls])
        idx += 1
        
    saveObject(img_ids, PATH.DATA.IMG_MAP)
    saveObject(img_cls_map, PATH.DATA.IMG_CLS_MAP)

    
'''
Main program

'''

if __name__=='__main__':
    datasets = CONFIG.DIS.DATASETS
    downloadDatasets(datasets)
    mapDatasets(datasets)

    # datasets = CONFIG.VIS.DATASETS
    # if IMAGENET in datasets:
    #     from utils.helper.imagenet_helper import *
    #     #downloadImageNet()
    #     mapImageNet()
