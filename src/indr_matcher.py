'''
Inner detector matcher:

To match semantic concepts with hidden units in intermediate layers

'''


import os, sys
from multiprocessing import Pool

curr_path = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.join(curr_path, "..")
if root_path not in sys.path:
    sys.path.insert(0, root_path)

from src.config import CONFIG, PATH

from utils.helper.data_loader import BatchLoader
from utils.helper.data_mapper import getClassName
from utils.helper.dstruct_helper import splitDict
from utils.helper.file_manager import saveObject, loadObject
from utils.model.model_agent import ModelAgent
from utils.dissection.activ_processor import reflect
from utils.dissection.identification import matchActivsAnnos, combineMatches

    
'''
Match annotaions and activation maps

'''

def reflectAndMatch(activ_maps, field_maps, annos):
    ref_activ_maps = reflect(activ_maps, field_maps)
    activ_maps = None

    batch_matches = matchActivsAnnos(ref_activ_maps, annos)
    ref_activ_maps = None

    return batch_matches


'''
Organise results and report

'''

def reportMatchResults(matches):
    print ("Report Matches: begin...")
    iou_thres = CONFIG.DIS.IOU_THRESHOLD
    top = CONFIG.DIS.TOP
    file_path = PATH.OUT.UNIT_MATCHES
    saveObject(matches, file_path)
    unit_matches = filterMatches(matches, top, iou_thres)

    concept_matches = rearrangeMatches(matches)
    file_path = PATH.OUT.CONCEPT_MATCHES
    saveObject(concept_matches, file_path)
    concept_matches = filterMatches(concept_matches, top, iou_thres)
    print ("Report Matches: filtering finished.")

    if CONFIG.DIS.REPORT_TEXT:
        file_path = PATH.OUT.UNIT_MATCH_REPORT
        reportMatchesInText(unit_matches, file_path, "unit")
        file_path = PATH.OUT.CONCEPT_MATCH_REPORT
        reportMatchesInText(concept_matches, file_path, "concept")
    print ("Report Matches: saved")
        
    if CONFIG.DIS.REPORT_FIGURE:
        reportMatchesInFigure(unit_matches)
    

def filterMatches(matches, top=3, iou_thres=0.00):
    filtered = {}
    for unit, unit_matches in matches.items():
        top_n = [None for x in range(top)]

        for concept, cct_match in unit_matches.items():
            idx = topIndex(top_n, cct_match[0])
            if idx is not None:
                top_n.insert(idx, (concept, cct_match[0]))
                top_n = top_n[:-1]

        retained = []
        for _match in top_n:
            if _match is None:
                break
            else:
                concept, iou = _match
            if iou >= iou_thres:
                unit_match = [concept] + unit_matches[concept]
                retained.append(unit_match)
        filtered[unit] = retained
    return filtered

def rearrangeMatches(matches):
    arranged = {}
    for unit, unit_matches in matches.items():
        for concept, cct_match in unit_matches.items():
            if concept not in arranged:
                unit_match = list(cct_match)
                arranged[concept] = {unit : unit_match}
            else:
                arranged[concept][unit] = list(cct_match)
    return arranged

def topIndex(top_n, iou):
    for idx, cct in enumerate(top_n):
        if cct is None:
            return idx
        elif iou > cct[1]:
            return idx
    return None
        
def reportMatchesInText(matches, file_path, form):
    with open(file_path, 'w+') as f:
        keys = list(matches.keys())
        keys.sort()
        for unit in keys:
            unit_matches = matches[unit]
            if form == "concept":
                unit = getClassName(unit, full=True)
            unit_line = "\n{}:\n".format(unit)
            f.write(unit_line)
            for match in unit_matches:
                concept = match[0]
                concept = getClassName(concept, full=True) if form=="unit" else concept
                iou = match[1]
                count = match[2]
                match_line = "{:20} \tIoU: {:.2f} \tCount: {:2}\n".format(concept, iou, count)
                f.write(match_line)
            if len(unit_matches) == 0:
                f.write("No significant matches found.\n")
    
def reportMatchesInFigure(matches):
    print ("placeholder")

    
'''
Main program

'''


if __name__ == "__main__":
    bl = BatchLoader(amount=100)
    probe_layers = loadObject(PATH.MODEL.PROBE)    
    model = ModelAgent()
    field_maps = model.getFieldmaps()

    pool = Pool()
    num = pool._processes
    matches = None
    while bl:
        batch = bl.nextBatch()
        images = batch[1]
        annos = batch[2]

        activ_maps = model.getActivMaps(images, probe_layers)
        activ_maps = splitDict(activ_maps, num)
        params = [(amap, field_maps, annos) for amap in activ_maps]
        print ("Reflecting and matching activation maps...")
        batch_matches = pool.starmap(reflectAndMatch, params)
        batch_match = {}
        for m in batch_matches:
            batch_match = {**batch_match, **m}
            
        print ("Integrating matches results of a batch into final results ...")
        matches = combineMatches(matches, batch_match)
        batch_matches = None
        
        bl.reportProgress()
        
    reportMatchResults(matches)
