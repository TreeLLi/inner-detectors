'''
Files manager

To manage all files' inputs and outpus

'''


import os, sys
import pickle
import json
import cv2

import matplotlib.pyplot as plt

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

def makeDirectory(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

        
'''
Images I/O

'''

def loadImage(directory, file_name, mode="BGR"):
    path = os.path.join(directory, file_name)
    try:
        img = cv2.imread(path)
        if img is None:
            raise Exception("Error: failed to load image {}".format(file_name))
        else:
            if mode == "RGB":
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return img
    except Exception as e:
        print (e)
        
def saveImage(img, file_path):
    try:
        makeDirectory(file_path)
        cv2.imwrite(file_path, img)
    except Exception as e:
        print (e)

def saveFigure(plt, file_path):
    try:
        makeDirectory(file_path)
        plt.savefig(file_path)
    except Exception as e:
        print (e)
        
        
'''
Objects I/O

'''
    
def loadObject(file_path, dstru='list'):
    try:
        ftype = file_path.split('.')[-1]
        if ftype == 'pkl':
            with open(file_path, 'rb') as f:
                obj = pickle.load(f, )
                return obj
        elif ftype == 'txt' and dstru == 'list':
            return loadListFromText(file_path)
        elif ftype == 'json':
            with open(file_path, encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        print ("Error: failed to load object from file {}".format(file_path))
        print ("Because {}".format(e))
        return None

def saveObject(obj, file_path):
    try:
        makeDirectory(file_path)
        ftype = file_path.split('.')[-1]
        if ftype == 'pkl':
            with open(file_path, 'wb+') as f:
                pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        elif ftype == 'txt' and isinstance(obj, list):
            saveListAsText(obj, file_path)
        elif ftype == 'json':
            with open(file_path, 'w+') as f:
                json.dump(obj, f)
    except Exception as e:
        print ("Error: failed to save object at {}".format(file_path))
        print ("Because {}".format(e))
        
        
'''
List-Text I/O

'''
        
def loadListFromText(file_path):
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
            for idx, line in enumerate(lines):
                line = line.rstrip("\n\t")
                splited = line.split(',')
                for _idx, e in enumerate(splited):
                    if e.isdigit():
                        splited[_idx] = int(e)
                splited = splited[0] if len(splited) == 1 else splited
                lines[idx] = splited
            return lines
    except Exception as e:
        print ("Error: failed to load text file {}".format(file_path))
        print ("Because {}".format(e))

def saveListAsText(obj, file_path):
    try:
        with open(file_path, 'w+') as f:
            for e in obj:
                if isinstance(e, list) or isinstance(e, tuple):
                    e = ["{:.2f}".format(x) if isinstance(x, float) else str(x)
                         for x in e]
                    line = ''.join([x if x is e[-1] else x+','
                                    for x in e])
                else:
                    line = str(e)
                line += '\n'
                f.write(line)
    except Exception as e:
        print ("Error: failed to save list at {}".format(file_path))
        print ("Because {}".format(e))

        
