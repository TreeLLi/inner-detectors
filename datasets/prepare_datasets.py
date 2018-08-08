'''
Data preparation

To download, organise and generate intermediate description files

'''


import sys
from operator import itemgetter
from os.path import exists, join, dirname, abspath
import numpy as np
import time

curr_path = dirname(abspath(__file__))
root_path = join(curr_path, "..")
if root_path not in sys.path:
    sys.path.insert(0, root_path)

from src.config import CONFIG, PATH, isPASCAL, isCOCO
from utils.helper.file_manager import loadObject, saveObject, getFilesInDirectory
from utils.helper.anno_parser import parsePASCALPartAnno
from utils.helper.data_mapper import mapClassID, getClassID, convert


'''
Download

'''

def downloadDatasets(sources):
    print ("download datasets of sources.")


'''
Images ID, Classes Mapping

'''

def mapDatasets(sources):
    maps = [[] for x in range(3)]
    maps[1].append("background")
    sources = sorted(sources)
    for source in sources:
        if isPASCAL(source):
            mapPASCAL(maps, source)
        elif isCOCO(source):
            mapCOCO(maps, source)
            
    maps[0] = [(idx, ) + _map for idx, _map in enumerate(maps[0])]
    paths = [PATH.DATA.IMG_MAP, PATH.DATA.CLS_MAP, PATH.DATA.IMG_CLS_MAP]
    for _map, _path in zip(maps, paths):
        saveObject(_map, _path)

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
        for img_id in data:
            img_cls = [img_id] + list(annos[img_id])
            img_cls_map.append(img_cls)
            img_ids.append((img_id, source, subset))

'''
Main program

'''

if __name__=='__main__':
    sources = CONFIG.DATA.SOURCES
    print ("Datasets preparation: downloading ...")
    downloadDatasets(sources)
    print ("Datasets preparation: download finished")
    print ("Datasets preparation: mapping images ...")
    mapDatasets(sources)
    print ("Datasets preparation: mapping finished")
    
