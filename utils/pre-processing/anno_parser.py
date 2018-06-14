import scipy.io as sio
import numpy as np
import os, pickle

from easydict import EasyDict as edict

'''
structures of annotations:

maps: {file_name : annos}

annos: {class_name : masks}

masks: {'class_id' : id of class,
        'class_mask' : mask of class object,
        'part_masks' : part_masks}

part_masks: [(part_name, part_mask), ...]

'''


'''
Helper functions
'''

def getFilesInDirectory(directory, postfix=""):
    file_names = [os.path.join(directory, f) for f in os.listdir(directory) if not os.path.isdir(os.path.join(directory, f))]
    if not postfix or postfix=="":
        return file_names
    else:
        return [f for f in file_names if f.lower().endswith(postfix)]

def _isMaskEmpty(mask):
    for row in mask:
        for column in row:
            if column != 0:
                return False
    return True

def save(name, data):
    with open(name, 'wb') as out:
        pickle.dump(data, out)

'''
Parse
'''

def parsePASCALPartAnno(mat_file):
    data = sio.loadmat(mat_file)
    data = data['anno']

    file_name = data[0][0][0][0]
    classes = edict()
    for cls in data[0][0][1][0]:
        cls_name = cls[0][0]
        cls_id = cls[1][0][0]
        cls_mask = cls[2]
    
        if _isMaskEmpty(cls_mask):
            print ("Exception: {} has no mask for class {}".format(file_name, cls_name))
            continue

        part_masks = []
        try:
            # access the list of part masks
            # if no part masks, cast exception and exit
            raw_part_masks = cls[3][0]
            for part in raw_part_masks:
                part_name = part[0][0]
                part_mask = part[1]
                if not _isMaskEmpty(part_mask):
                    part_masks.append((part_name, part_mask))
            if part_masks:
                # store elements if effective part masks exist
                classes[cls_name] = edict({
                    "class_id" : cls_id,
                    "class_mask" : cls_mask,
                    "part_masks" : part_masks})
        except:
            print ("Exception: {} has no part masks for {}".format(file_name, cls_name))
        
    return file_name, classes
    
def parsePASCALPartAnnos(directory, postfix=""):
    files = getFilesInDirectory(directory, postfix)
    
    maps = {}
    count = 0
    for f in files:
        print (f)
        file_name, masks = parsePASCALPartAnno(f)
        if not masks:
            # skip this sample, if no mask for the class or parts
            continue
        else:
            count += 1
            
        maps[file_name] = masks

        if count == 10:
            # save small samples for testing
            save("pascal_annos_{}.pkl".format(count), maps)
            
    save("pascal_annos_all.pkl", maps)

    
'''
Testing
'''

if __name__ == "__main__":
    directory = "Annotations_Part"
    postfix = 'mat'

    parsePASCALPartAnnos(directory, postfix)

    # file_name, masks = parsePASCALPartAnno(directory+"/2008_001862.mat")
    # for cls, mask in masks.items():
    #     print (cls)
        
