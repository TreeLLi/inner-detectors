'''
Inner detector matcher:

To match semantic concepts with hidden units in intermediate layers

'''


import os, sys
import time
from easydict import EasyDict as edict
from multiprocessing import Pool

curr_path = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.join(curr_path, "..")
if root_path not in sys.path:
    sys.path.insert(0, root_path)

from utils.helper.data_loader import BatchLoader
from utils.helper.file_manager import saveObject, loadObject
from utils.model.model_agent import ModelAgent
from src.config import CONFIG, PATH, isModeFast
from utils.dissection.iou import iou
    

'''
Fast Mode Functions
'''

def loadRefActivMaps(img_names, layers):
    ref_activ_maps = {}
    for name in img_names:
        file_path = os.path.join(PATH.MODEL.REF_ACTIV_MAPS, name+'.pkl')
        ref_activ_map = loadObject(file_path)
        ref_activ_map = retainLayers(ref_activ_map, layers)

        for unit_id, ramap in ref_activ_map.items():
            if unit_id not in ref_activ_maps:
                ref_activ_maps[unit_id] = [ramap]
            else:
                ref_activ_maps[unit_id].append(ramap)
    return ref_activ_maps

def retainLayers(ramap, layers):
    retained = {}
    for layer in layers:
        exist = False
        for unit_id, m in ramap.items():
            if layer in unit_id:
                retained[unit_id] = m
                exist = True
        if not exist:
            raise Exception("Exception: fast mode - incompleted stored data lacking layer {}"
                            .format(layer))
    return retained

def saveRefActivMaps(ref_activ_maps, names):
    for idx, name in enumerate(names):
        ramap = {}
        for unit_id, ramaps in ref_activ_maps.items():
            ramap[unit_id] = ramaps[idx]
        file_path = os.path.join(PATH.MODEL.REF_ACTIV_MAPS, name+'.pkl')
        saveObject(ramap, file_path)

    
'''
Match annotaions and activation maps

'''

# match activation maps of all units, of a batch of images,
# with all annotations of corresponding images
def matchActivsAnnos(activs, annos):
    matches = edict()
    for unit, activs in activs.items():
        unit_matches = []
        for img_idx, activ in enumerate(activs):
            img_annos = annos[img_idx]
            unit_img_matches = matchActivAnnos(activ, img_annos)
            unit_matches.append(unit_img_matches)
        matches[unit] = unit_matches
        # print ("Matched: unit {}".format(unit))

    return matches


# match a single activation map with annotations from one image
def matchActivAnnos(activ, annos):
    matches = edict()
    for anno in annos:
        concept = anno.name
        category = anno.category
        mask = anno.mask

        if concept not in matches:
            matches[concept] = edict()
            matches[concept].category = category
            matches[concept].iou = iou(activ, mask)
            matches[concept].count = 1
        else:
            # multiple same concepts exist in the same image
            iou_1 = matches[concept].iou
            count_1 = matches[concept].count
            iou_2 = iou(activ, mask)
            matches[concept].iou = weightedIoU(iou_1, count_1, iou_2)
            matches[concept].count += 1

        # print ("\tConcept:{:6}\tIoU:{:.2f}\tCount:{:2}".format(concept, matches[concept].iou, matches[concept].count))
    return matches


# calculate count-weighted IoU
def weightedIoU(iou_1, count_1, iou_2, count_2=1):
    return (iou_1*count_1 + iou_2*count_2) / (count_1+count_2)


# combine two matches results
def combineMatches(matches, batch_matches):
    if batch_matches is None:
        return matches
    elif matches is None:
        matches = edict()
        
    for unit, batch_match in batch_matches.items():
        if unit not in matches:
            matches[unit] = edict()

        unit_match = matches[unit]
        for img_match in batch_match:
            for concept, cct_match in img_match.items():
                if concept not in unit_match:
                    unit_match[concept] = cct_match
                else:
                    iou_1 = cct_match.iou
                    count_1 = cct_match.count
                    iou_2 = unit_match[concept].iou
                    count_2 = unit_match[concept].count
                    unit_match[concept].iou = weightedIoU(iou_1, count_1, iou_2, count_2)
                    unit_match[concept].count += count_1
    return matches


'''
Organise results and report

'''

def reportMatchResults(matches):
    print ("Report Matches: begin...")
    iou_thres = CONFIG.DIS.IOU_THRESHOLD
    top = CONFIG.DIS.TOP
    unit_matches = filterMatches(matches, top, iou_thres)
    concept_matches = rearrangeMatches(matches, top, iou_thres)
    concept_matches = filterMatches(concept_matches, top, iou_thres)
    print ("Report Matches: filtering finished.")

    # save object files for further probe
    file_path = PATH.OUT.UNIT_MATCHES
    saveObject(unit_matches, file_path)
    file_path = PATH.OUT.CONCEPT_MATCHES
    saveObject(concept_matches, file_path)
    
    if CONFIG.DIS.REPORT_TEXT:
        file_path = PATH.OUT.UNIT_MATCH_REPORT
        reportMatchesInText(unit_matches, file_path)
        file_path = PATH.OUT.CONCEPT_MATCH_REPORT
        reportMatchesInText(concept_matches, file_path)
    print ("Report Matches: saved")
        
    if CONFIG.DIS.REPORT_FIGURE:
        reportMatchesInFigure(unit_matches)
    

def filterMatches(matches, top=3, iou_thres=0.00):
    filtered = {}
    for unit, unit_matches in matches.items():
        top_n = [None for x in range(top)]

        for concept, cct_match in unit_matches.items():
            idx = topIndex(top_n, cct_match.iou)
            if idx is not None:
                top_n.insert(idx, (concept, cct_match.iou))
                top_n = top_n[:-1]

        retained = []
        for concept, iou in top_n:
            if iou >= iou_thres:
                unit_match = edict(unit_matches[concept])
                unit_match.name = concept
                retained.append(unit_match)
        filtered[unit] = retained
    return filtered

def rearrangeMatches(matches, top, iou_thres):
    arranged = {}
    for unit, unit_matches in matches.items():
        for concept, cct_match in unit_matches.items():
            if concept not in arranged:
                unit_match = edict(cct_match)
                arranged[concept] = {unit : unit_match}
            else:
                arranged[concept][unit] = edict(cct_match)
    return arranged

def topIndex(top_n, iou):
    for idx, cct in enumerate(top_n):
        if cct is None:
            return idx
        elif iou > cct[1]:
            return idx
    return None
        
def reportMatchesInText(matches, file_path):
    model = CONFIG.DIS.MODEL
    
    with open(file_path, 'w') as f:
        for unit, unit_matches in matches.items():
            unit_line = "\n{}:\n".format(unit)
            f.write(unit_line)
            for match in unit_matches:
                match_line = "{:10} \tIoU: {:.2f} \tCount: {:2}\n".format(match.name,
                                                                          match.iou,
                                                                          match.count)
                f.write(match_line)
            if len(unit_matches) == 0:
                f.write("No significant matches found.\n")
        
def reportMatchesInFigure(matches):
    print ("placeholder")


'''
Progress report

'''

def reportProgress(start, end, bid, num):
    dur = end - start
    effi = dur / num
    print ("Batch {}: finished {} samples in {:.2f} sec. ({:.2f} sec. / sample)"
           .format(bid, num, dur, effi))
    

'''
Main program

'''


def reflectAndMatch(activ_maps, field_maps, annos):
    print ("Mapping activation maps back to input images ...")
    ref_activ_maps = reflect(activ_maps, field_maps, annos)
    activ_maps = None

    print ("Matching activations and annotations ...")
    batch_matches = matchActivsAnnos(ref_activ_maps, annos)
    ref_activ_maps = None

    return batch_matches


def splitActivMaps(activ_maps, num):
    keys = list(activ_maps.keys())
    size = len(keys) // num
    left = len(keys) % num
    sizes = [size for x in range(num)]
    for i in range(num):
        sizes[i] += 1 if left>0 else 0
        left -= 1
    splited = []
    for size in sizes:
        sub_keys = keys[:size]
        splited.append({key:activ_maps[key] for key in sub_keys})
        keys = keys[size:]
    return splited

if __name__ == "__main__":
    bl = BatchLoader(amount=30)
    model = ModelAgent(input_size=10)
    probe_layers = loadObject(PATH.MODEL.PROBE)

    if CONFIG.DIS.REFLECT == "interpolation":
        from utils.dissection.interp_ref import reflect
        field_maps = model.getFieldmaps()
    elif CONFIG.DIS.REFLECT == "deconvnet":
        from utils.dissection.deconvnet import reflect

    pool = Pool()
    num = pool._processes
    matches = None
    while bl:
        start = time.time()
        batch = bl.nextBatch()
        names = batch.names
        images = batch.imgs
        annos = batch.annos

        print ("Fetching activation maps for specific units ...")
        activ_maps = model.getActivMaps(images, probe_layers)
        activ_maps = splitActivMaps(activ_maps, num)
        params = [(amap, field_maps, annos) for amap in activ_maps]
        batch_matches = pool.starmap(reflectAndMatch, params)
        
        print ("Integrating matches results of a batch into final results ...")
        for batch_match in batch_matches:
            matches = combineMatches(matches, batch_match)
        batch_matches = None
        
        reportProgress(start, time.time(), bl.batch_id, len(images))
        
    reportMatchResults(matches)
