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

def _isMaskEmpty(mask):
    for row in mask:
        for column in row:
            if column != 0:
                return False
    return True

# def save(name, data):
#     with open(name, 'wb') as out:
#         pickle.dump(data, out)

'''
Parse
'''

def parsePASCALPartAnno(directory, file_name):
    path = os.path.join(directory, file_name)
    data = sio.loadmat(path)
    data = data['anno']

    file_id = data[0][0][0][0]
    annos = []
    labels = []
    for cls in data[0][0][1][0]:
        cls_name = cls[0][0]
        cls_id = cls[1][0][0]
        cls_mask = cls[2]
    
        if _isMaskEmpty(cls_mask):
            print ("Exception: {} has no mask for class {}".format(file_id, cls_name))
            continue
        else:
            labels.append(cls_id)
            annos.append(edict({
                "name" : cls_name,
                "category" : "object",
                "mask" : cls_mask}))
            
        try:
            # access the list of part masks
            # if no part masks, cast exception and exit
            part_masks = cls[3][0]
            for part in part_masks:
                part_name = part[0][0]
                part_mask = part[1]
                if not _isMaskEmpty(part_mask):
                    annos.append(edict({
                        "name" : part_name,
                        "category" : "part",
                        "mask" : part_mask,
                        "partof" : cls_name}))
        except:
            print ("Exception: {} has no part masks for {}".format(file_id, cls_name))
        
    return annos, labels
    
# def parsePASCALPartAnnos(directory, postfix=""):
#     files = getFilesInDirectory(directory, postfix)
    
#     maps = {}
#     count = 0
#     for f in files:
#         print (f)
#         file_id, masks = parsePASCALPartAnno(f)
#         if not masks:
#             # skip this sample, if no mask for the class or parts
#             continue
#         else:
#             count += 1
            
#         maps[file_id] = masks

#         if count == 10:
#             # save small samples for testing
#             save("pascal_annos_{}.pkl".format(count), maps)
            
#     save("pascal_annos_all.pkl", maps)

    
'''
Testing
'''

# if __name__ == "__main__":
#     directory = "Annotations_Part"
#     postfix = 'mat'

#     parsePASCALPartAnnos(directory, postfix)

#     # file_id, masks = parsePASCALPartAnno(directory+"/2008_001862.mat")
#     # for cls, mask in masks.items():
#     #     print (cls)
        
