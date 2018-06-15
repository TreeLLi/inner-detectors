'''
Data loader:

Manage the batch loading of data concerning different purposes and datasets

Batch structure:
-TODO:

'''

from easydict import EasyDict as edict
import os, sys

curr_path = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.join(curr_path, "../..")
sys.path.append(root_path)

from src.config import PATH
from utils.helper.file_manager import *
from utils.helper.anno_parser import parsePASCALPartAnno


# global constants for specify the dataset sources
PASCAL = "PASCAL"
SOURCE = [PASCAL]

'''
dataset-specific loading functions

'''

def loadPASCALDataList():
    directory = PATH.DATA.PASCAL.ANNOS
    postfix = "mat"
    data = getFilesInDirectory(directory, postfix)
    return [(x[x.rfind('/')+1:-4], PASCAL) for x in data]
    
def fetchDataFromPASCAL(identifier):
    img_postfix = ".jpg"
    anno_postfix = ".mat"
    img_dir = PATH.DATA.PASCAL.IMGS
    anno_dir = PATH.DATA.PASCAL.ANNOS

    img = loadImage(img_dir, identifier+img_postfix)
    annos, labels = parsePASCALPartAnno(anno_dir, identifier+anno_postfix)

    return img, annos, labels


'''
Load data as batches

'''

class BatchLoader(object):

    def __init__(self, sources=[PASCAL], batch_size=10, amount=None):
        self.batch_size = batch_size
        
        self.data = []
        for source in sources:
            if source not in SOURCE:
                continue
            elif source == PASCAL:
                self.data += loadPASCALDataList()

            if amount is not None and len(self.data)>amount:
                # discard remaining datasets, since the specified amount is reached
                self.data = self.data[:amount]
                break

        self.amount = amount if amount is not None else len(self.data)
        
    def __bool__(self):
        return len(self.data) != 0

    @property
    def size(self):
        return len(self.data)
    
    def nextBatch(self, amount=None):
        # adjust the amount of next batch based on the specified amount
        # and the actual remaining number of data
        num = self.batch_size if amount is None else amount
        num = num if self.size > num else self.size

        samples = self.data[:num]
        self.data = self.data[num:]

        batch = edict({
            "ids" : [],
            "imgs" : [],
            "annos" : [],
            "labels" : []
        })
        
        for sample in samples:
            s_id = sample[0]
            s_source = sample[1]

            if s_source == PASCAL:
                img, annos, labels = fetchDataFromPASCAL(s_id)
            # add operations for other sources

            batch.ids.append(s_id)
            batch.imgs.append(img)
            batch.annos.append(annos)
            batch.labels.append(labels)
            
        return batch
