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
        combineMatches(matches, batch_matches)

    saveMatches(matches)


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
    if matches is None or batch_matches is None:
        return matches if batch_matches is None else batch_matches

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
        

    
