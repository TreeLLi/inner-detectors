'''
Files manager

To manage all files' inputs and outpus

'''


import os, sys
import pickle
from scipy.misc import imread


'''
General-purpose Auxiliary Func

'''

def getFilesInDirectory(directory, postfix=""):
    file_names = [os.path.join(directory, f) for f in os.listdir(directory)
                  if not os.path.isdir(os.path.join(directory, f))]
    if not postfix or postfix=="":
        return file_names
    else:
        return [f for f in file_names if f.lower().endswith(postfix)]

    
'''
Images I/O

'''

def loadImage(directory, file_name):
    path = os.path.join(directory, file_name)
    try:
        img = imread(path)
        return img
    except:
        print ("Error: failed to load image {}".format(file_name))
        return None

    
'''
Objects I/O

'''
    
def loadObject(file_path, dstru='list'):
    try:
        ftype = file_path.split('.')[-1]
        if ftype == 'pkl':
            with open(file_path, 'rb') as f:
                obj = pickle.load(f)
                return obj
        elif ftype == 'txt' and dstru == 'list':
            return loadListFromText(file_path)
    except:
        print ("Error: failed to load object from file {}".format(file_path))
        return None

def saveObject(obj, file_path):
    try:
        ftype = file_path.split('.')[-1]
        if ftype == 'pkl':
            with open(file_path, 'wb+') as f:
                pickle.dump(obj, f)
        elif ftype == 'txt' and isinstance(obj, list):
            saveListAsText(obj, file_path)
    except:
        print ("Error: failed to save object at {}".format(file_path))

        
'''
List-Text I/O

'''
        
def loadListFromText(file_path):
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
            lines = [line.rstrip('\n') for line in lines]
            lines = [line.split(',') for line in lines]
            lines = [line[0] if len(line)==1 else line for line in lines]
            return lines
    except:
        print ("Error: failed to load text file {}".format(file_path))
        return None

def saveListAsText(obj, file_path):
    try:
        with open(file_path, 'w+') as f:
            for e in obj:
                if isinstance(e, list) or isinstance(e, tuple):
                    line = ''.join([str(x) if x is e[-1] else str(x)+','
                                    for x in e])
                else:
                    line = str(e)
                line += '\n'
                f.write(line)
    except:
        print ("Error: failed to save list at {}".format(file_path))
