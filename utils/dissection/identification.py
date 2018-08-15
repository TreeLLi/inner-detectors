import os, sys
import numpy as np
import time

curr_path = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.join(curr_path, "..")
if root_path not in sys.path:
    sys.path.insert(0, root_path)

from src.config import PATH
    
from utils.dissection.helper import iou
from utils.model.model_agent import isUnitID, splitUnitID
from utils.helper.data_loader import weightedVal
from utils.helper.dstruct_helper import nested, sortDict, reverseDict
from utils.helper.file_manager import loadObject


'''
Identification results

'''

def identification(matches=None, top=None, mode='unit', organise=False):
    start = time.time()
    
    if matches is None:
        if mode == 'unit':
            data_path = PATH.OUT.IDE.DATA.UNIT
        elif mode == 'concept':
            data_path = PATH.OUT.IDE.DATA.CONCEPT
        else:
            raise Exception("Error: invalid mode for loading identification")
        
        print ("Identification: loaded and processed from matches data.")
        if os.path.exists(data_path):
            matches = loadObject(data_path)
        else:
            raise Exception("Error: no data for identification")

    if mode == 'unit' and isConceptForm(matches):
        print ("Matches: convert from units to concepts.")
        matches = reverseDict(matches)
    elif mode == 'concept' and isUnitForm(matches):
        print ("Matches: convert from concepts to units.")
        matches = reverseDict(matches)
    
    # sorting
    if mode=='concept' and organise:
        matches = organiseMatches(matches, top)
    elif top:
        matches = {k : sortDict(v, indices=[0], merge=True)[:top] for k, v in matches.items()}
    else:
        matches = {k : sortDict(v, indices=[0], merge=True) for k, v in matches.items()}

    end = time.time()
    print ("Identification: finished {}s.".format(int(end-start)))
    return matches

def isUnitForm(matches):
    return any(isUnitID(uid) for uid in matches.keys())

def isConceptForm(matches):
    return not isUnitForm(matches)

def organiseMatches(matches, top=None):
    organised = {}
    for ccp, unit_id, m in nested(matches, depth=2):
        if ccp not in organised:
            organised[ccp] = {}

        layer, unit = splitUnitID(unit_id)
        if layer not in organised[ccp]:
            organised[ccp][layer] = {}
        organised[ccp][layer][unit] = m

    for ccp, layer, units in nested(organised, depth=2):
        if top:
            organised[ccp][layer] = sortDict(units, indices=[0], merge=True)[:top]
        else:
            organised[ccp][layer] = sortDict(units, indices=[0], merge=True)

    return organised


'''
Match activation and concepts

'''

# match activation maps of all units, of a batch of images,
# with all annotations of corresponding images
def matchActivsAnnos(activs, annos):
    matches = {}
    for unit, activs in activs.items():
        unit_matches = []
        for img_idx, activ in enumerate(activs):
            img_annos = annos[img_idx]
            unit_img_matches = matchActivAnnos(activ, img_annos)
            unit_matches.append(unit_img_matches)
        matches[unit] = unit_matches

    return matches


# match a single activation map with annotations from one image
def matchActivAnnos(activ, annos):
    matches = {}
    for anno in annos:
        concept = anno[0]
        mask = anno[1]

        if concept not in matches:
            matches[concept] = [iou(activ, mask), 1]
        else:
            # multiple same concepts exist in the same image
            iou_1 = matches[concept][0]
            count_1 = matches[concept][1]
            iou_2 = iou(activ, mask)
            matches[concept][0] = weightedVal(iou_1, count_1, iou_2)
            matches[concept][1] += 1

    return matches

# combine two matches results
def combineMatches(matches, batch_matches):
    if batch_matches is None:
        return matches
    elif matches is None:
        matches = {}
        
    for unit, batch_match in batch_matches.items():
        if unit not in matches:
            matches[unit] = {}

        unit_match = matches[unit]
        for img_match in batch_match:
            for concept, cct_match in img_match.items():
                if concept not in unit_match:
                    unit_match[concept] = cct_match
                else:
                    iou_1 = cct_match[0]
                    count_1 = cct_match[1]
                    iou_2 = unit_match[concept][0]
                    count_2 = unit_match[concept][1]
                    unit_match[concept][0] = weightedVal(iou_1, count_1, iou_2, count_2)
                    unit_match[concept][1] += count_1
    return matches
