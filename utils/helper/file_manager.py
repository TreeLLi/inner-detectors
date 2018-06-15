'''
Files manager

To manage all files' inputs and outpus

'''


import os, sys
from scipy.misc import imread


def getFilesInDirectory(directory, postfix=""):
    file_names = [os.path.join(directory, f) for f in os.listdir(directory) if not os.path.isdir(os.path.join(directory, f))]
    if not postfix or postfix=="":
        return file_names
    else:
        return [f for f in file_names if f.lower().endswith(postfix)]


def loadImage(directory, file_name):
    path = os.path.join(directory, file_name)
    try:
        img = imread(path)
        return img
    except:
        print ("Error: failed to load image {}".format(file_name))
        return None

def loadBinaryData(path):
    try:
        with open(path, 'rb') as data_file:
            data = pickle.load(data_file)
            return data
    except:
        print ("Error: failed to load data {}".format(path))
        return None

