'''
Inner detector matcher:

To match semantic concepts with hidden units in intermediate layers

'''


import os, sys
from easydict import EasyDict as edict

curr_path = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.join(curr_path, "..")
if root_path not in sys.path:
    sys.path.insert(0, root_path)

from utils.helper.data_loader import BatchLoader
from utils.model.model_agent import ModelAgent
from src.config import CONFIG
from utils.dissection.iou import iou

'''
Main program

'''


if __name__ == "__main__":
    bl = BatchLoader()
    matches = None
    
    while bl:
        batch = bl.nextBatch()
        images = batch.imgs

        model = ModelAgent()
        activ_maps = model.getActivMaps(images)
        
        if CONFIG.DIS.REFLECT == "linear":
            from utils.dissection.linear_ref import reflect
        elif CONFIG.DIS.REFLECT == "deconvnet":
            from utils.dissection.deconvnet import reflect

        ref_activ_maps = reflect(activ_maps, model=CONFIG.DIS.MODEL)
        annos = batch.annos

        batch_matches = matchActivsAnnos(ref_activ_maps, annos)
        matches = combineMatches(matches, batch_matches)

    reportMatchResults(matches)


'''
Match annotaions and activation maps

'''

# match activation maps of all units, of a batch of images,
# with all annotations of corresponding images
def matchActivsAnnos(activs, annos):
    matches = edict
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
                print (cct_match)
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
    retained_matches = filterMatches(matches, top, iou_thres)
    print ("Report Matches: finish filtering.")
    
    if CONFIG.DIS.REPORT_TEXT:
        reportMatchesInText(retained_matches)

    if CONFIG.DIS.REPORT_FIGURE:
        reportMatchesInFigure(retained_matches)
        

def filterMatches(matches, top=1, iou_thres=0.3):
    for unit, unit_matches in matches.items():
        top_n = [None for x in range(top)]

        for concept, cct_match in unit_matches.items():
            idx = topIndex(top_n, cct_match.iou)
            if idx < top:
                top_n[idx] = (concept, cct_match.iou)

        retained = []
        for concept, cct_match in top_n:
            if cct_match.iou >= iou_thres:
                match = [concept, cct_match.iou, cct_match.count]
                retained.append(match)
        matches[unit] = retained

        
def reportMatchesInText(matches):
    model = CONFIG.DIS.MODEL
    out_path = PATH.OUT.MATCH
    file_path = "{}{}_matches.txt".format(out_path, model)
    
    with open(file_path, 'w') as f:
        for unit, unit_matches in matches.items():
            unit_line = "{}:\n".format(unit)
            f.write(unit_line)
            
            for match in unit_matches:
                match_line = "{:10} IoU: {:.2f} Count: {:3}".format(match[0], match[1], match[2])
                f.write(match_line)


def reportMatchesInFigure(matches):
    print ("placeholder")
