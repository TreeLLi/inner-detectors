'''
Data preparation

To download, organise and generate intermediate description files

'''


import os, sys
from operator import itemgetter

curr_path = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.join(curr_path, "..")
if root_path not in sys.path:
    sys.path.insert(0, root_path)

from src.config import CONFIG, PATH, isPASCAL
from utils.helper.file_manager import saveObject, getFilesInDirectory


'''
Download

'''

def downloadDatasets(sources):
    print ("download datasets of sources.")

    
'''
Images id mapping

'''

def mapDatasets(sources):
    maps = []
    # sort sources to guarantee same order for same sources
    sources = sorted(sources)
    for source in sources:
        if isPASCAL(source):
            _maps = loadPASCALDataList(source)
            _maps = sorted(_maps, key=itemgetter(0))
        maps += _maps
    maps = [x + (idx,) for idx, x in enumerate(maps)]
    saveObject(maps, PATH.DATA.MAPS)

def loadPASCALDataList(source_id):
    directory = PATH.DATA.PASCAL.ANNOS
    postfix = "mat"
    data = getFilesInDirectory(directory, postfix)
    return [(x[x.rfind('/')+1:-4], source_id) for x in data]
    

'''
Generate statistical description of datasets

'''

def describeDatasets(sources):
    print ("describe datasets")


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
    print ("Datasets preparation: generating description ...")
    describeDatasets(sources)
    print ("Datasets preparation: descrition finished")
