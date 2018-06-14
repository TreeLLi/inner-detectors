'''
Data loader:

Manage the batch loading of data concerning different purposes and datasets

'''

from easydict import EasyDict as edict
from scipy.misc import imread

import pickle, os

SOURCE = edict({
    "PASCAL" : "PASCAL Part"})


def loadBinaryData(path):
    try:
        with open(path, 'rb') as data_file:
            data = pickle.load(data_file)
            return data
    except:
        print ("Error: failed to load data {}".format(path))
        return None

def loadImages(directory, names):
    images = editc()
    for name in names:
        path = os.path.join(directory, name)
        try:
            images[name] = imread(path)
        except:
            print ("Error: failed to load image {}".format(name))

    return images



class BatchLoader:

    def __init__(self, batch_size=10, capacity=None, sources):
        self.batch_size = batch_size
        self.capacity = capacity
        
        self.data = edict({"annos" : edict(), "imgs" : edict()})
        for source in sources:
            if source not in SOURCE:
                continue

            # load data from source
            annos = loadBinaryData("datasets/annos_10.pkl")
            imgs = loadImages("datasets/VOC2010/JPEGImages/", annos.keys())
            self.data.annos.update(annos)
            self.data.imgs.update(imgs)
        
    def __bool__(self):
        return self.data.imgs

    def next(self, amount=None):
        
