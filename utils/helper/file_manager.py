'''
Files manager

To manage all files' inputs and outpus

'''


import os, sys
import pickle
import json
import cv2


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

def loadImage(directory, file_name, mode="BGR"):
    path = os.path.join(directory, file_name)
    try:
        img = cv2.imread(path)
        if img is None:
            raise Exception()
        else:
            if mode == "RGB":
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return img
    except:
        print ("Error: failed to load image {}".format(file_name))
        return None

def saveImage(img, file_path):
    try:
        directory = os.path.dirname(file_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        cv2.imwrite(file_path, img)
    except Exception as e:
        print (e)
        print ("Error: failed to save image at {}".format(directory))
    
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
        directory = os.path.dirname(file_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
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
            lines = [line.rstrip("\n\t") for line in lines]
            lines = [line.split(',') for line in lines]
            lines = [line[0] if len(line)==1 else line for line in lines]
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
    
