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
from config import CONFIG


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

        matches.concept = edict()
        matches.concept.category = category
        matches.concept.iou = iou(activ, mask)
        
    return matches
