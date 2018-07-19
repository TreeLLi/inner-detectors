'''
Data preparation

To download, organise and generate intermediate description files

'''


import os, sys
from operator import itemgetter
import numpy as np

curr_path = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.join(curr_path, "..")
if root_path not in sys.path:
    sys.path.insert(0, root_path)

from src.config import CONFIG, PATH, isPASCAL
from utils.helper.file_manager import saveObject, getFilesInDirectory
from utils.helper.anno_parser import parsePASCALPartAnno
from utils.helper.data_loader import mapClassId


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
    saveObject(maps, PATH.DATA.IMG_MAP)
    return maps

def loadPASCALDataList(source_id):
    directory = PATH.DATA.PASCAL.ANNOS
    postfix = "mat"
    data = getFilesInDirectory(directory, postfix)
    return [(x[x.rfind('/')+1:-4], source_id) for x in data]
    

'''
Generate statistical description of datasets

'''


# def describeDatasets(maps):
#     mappings = [[], []]
#     des = {}
#     for img_id, source, _ in maps:
#         if isPASCAL(source):
#             directory = PATH.DATA.PASCAL.ANNOS
#             file_name = img_id + ".mat"
#             annos = parsePASCALPartAnno(directory, file_name, mappings, mapClassId)
#         for anno in annos:
#             cls_id, cls_mask = anno
#             cls_des = des[cls_id] if cls_id in des else [0 for x in range(8)]
#             updateCLassDes(cls_des, cls_mask)

# def updateClassDes(des, mask):
#     row_num = [np.sum(row[row>0]) for row in mask]
#     row_num = row_num[row_num>0]
#     col_num = [mask[:, i] for i in mask.shape[1]]
#     col_num = [np.sum(col[col>0]) for col in col_num]
#     col_num = col_num[col_num>0]
#     c_1 = des[0]
    
#     ops = [min, max, np.mean]
#     vals = [op(num) for op, num in zip(ops, [row_num, col_num])]
#     for idx, val in enumerate(des[1:-1]):
#         des[idx+1] = weightedVal(val, c_1, vals[idx])
#     des[-1] = weightedVal(des[-1], c_1, np.sum(mask[mask>0]))
#     des[0] += 1
    
# def weightedVal(v_1, c_1, v_2, c_2=1):
#     return (v_1*c_1 + v_2*c_2) / (c_1 + c_2)


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
    # if CONFIG.DATA.STATISTICS:
    #     print ("Datasets preparation: generating description ...")
    #     describeDatasets(maps)
    #     print ("Datasets preparation: descrition finished")
