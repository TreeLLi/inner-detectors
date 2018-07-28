'''
Data preparation

To download, organise and generate intermediate description files

'''


import os, sys
from operator import itemgetter
from os.path import exists
import numpy as np
import time

curr_path = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.join(curr_path, "..")
if root_path not in sys.path:
    sys.path.insert(0, root_path)

from src.config import CONFIG, PATH, isPASCAL
from utils.helper.file_manager import loadObject, saveObject, getFilesInDirectory
from utils.helper.anno_parser import parsePASCALPartAnno
from utils.helper.data_loader import mapClassID


'''
Download

'''

def downloadDatasets(sources):
    print ("download datasets of sources.")


'''
Images id mapping

'''

def mapDatasets(sources):
    path = PATH.DATA.IMG_MAP
    if not os.path.exists(path):
        img_ids = mapImageID(sources)
        saveObject(maps, path)
    else:
        img_ids = loadObject(path)

    
    mappings = [None, None]
    path = PATH.DATA.CLS_MAP
    mappings[0] = loadObject(path) if exists(path) else []
    path = PATH.DATA.IMG_CLS_MAP
    mappings[1] = None if (exists(path) and loadObject(path)) else []

    amount = len(img_ids)
    start = time.time()
    for idx, img_id in enumerate(img_ids):
        if idx % 100 == 0:
            remg = amount - idx
            per = remg / float(amount) * 100
            dur = time.time() - start
            start = time.time()
            eff = dur / 100
            print ("Mapping class: finished {}, remaining {}({:.2f}%), effi {:.2f}/img"
                   .format(idx, remg, per, eff))
            
        if isPASCAL(img_id[1]):
            dirt = PATH.DATA.PASCAL.ANNOS
            file_name = img_id[0] + ".mat"
            parsePASCALPartAnno(dirt, file_name, mappings, mapClassID)
        
    if not exists(path):
        saveObject(mappings[1], path)

    path = PATH.DATA.CLS_MAP
    if not exists(path):
        saveObject(mappings[0], path)


'''
Map image id

'''


def mapImageID(sources):
    maps = []
    # sort sources to guarantee same order for same sources
    sources = sorted(sources)
    for source in sources:
        if isPASCAL(source):
            _maps = loadPASCALDataList(source)
            _maps = sorted(_maps, key=itemgetter(0))
        maps += _maps
    maps = [x + (idx,) for idx, x in enumerate(maps)]
    return maps

def loadPASCALDataList(source_id):
    directory = PATH.DATA.PASCAL.ANNOS
    postfix = "mat"
    data = getFilesInDirectory(directory, postfix)
    return [(x[x.rfind('/')+1:-4], source_id) for x in data]


'''
Main program

'''

if __name__=='__main__':
    sources = CONFIG.DATA.SOURCES
    print ("Datasets preparation: downloading ...")
    downloadDatasets(sources)
    print ("Datasets preparation: download finished")
    print ("Datasets preparation: mapping images ...")
    maps = mapDatasets(sources)
    print ("Datasets preparation: mapping finished")
    
