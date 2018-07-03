'''
Files manager

To manage all files' inputs and outpus

'''


import os, sys
import pickle
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

def loadListFromText(file_path):
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
            return [line.rstrip('\n') for line in lines]
    except:
        print ("Error: failed to load text file {}".format(file_path))
        return None

def loadObject(file_path):
    try:
        with open(file_path, 'r') as f:
            obj = pickle.load(f)
            return obj
    except:
        print ("Error: failed to load object from file {}".format(file_path))
        return None

def saveObject(obj, file_path):
    try:
        with open(file_path, 'wb+') as f:
            pickle.dump(obj, f)
    except:
        print ("Error: failed to save object at {}".format(file_path))
