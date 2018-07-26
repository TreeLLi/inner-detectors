'''
Inner detectors verifier

To verify the semantic association

'''


import os, sys
from itertools import product

curr_path = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.join(curr_path, "..")
if root_path not in sys.path:
    sys.path.insert(0, root_path)

from utils.helper.data_loader import BatchLoader
from utils.helper.data_processor import activAttrs
from utils.helper.file_manager import saveObject, loadObject


'''
Activation attributes processing

'''

def updateActivAttrDiffs(attr_diffs, activ_attrs, anno_ids, patched=False):
    update_idx = 1 if patched else 0
    for unit_id, unit_activ_attrs in activ_attrs.items():
        for img_activ_attrs, aid in product(unit_activ_attrs, anno_ids):
            if unit_id not in attr_diffs:
                attr_diffs[unit_id] = {aid : []}
            diffs = attr_diffs[unit_id]

            if aid not in diffs:
                diffs[aid] = []
            diffs = diffs[aid]
            
            for attr_idx, attr in enumerate(img_activ_attrs):
                if attr_idx >= len(diffs):
                    diffs += [[0,0]] * (attr_idx+1 - len(diffs))
                diffs[attr_idx][update_idx] += attr
            

'''
Main program

'''


if __name__ == "__main__":
    bl = BatchLoader(amout=10)
    model = ModelAgent()
    probe_layers = loadObject(PATH.MODEL.PROBE)

    attr_diffs = {}
    while bl:
        batch = bl.nextBatch()
        imgs = batch[1]
        annos = batch[2]

        activ_maps = model.getAcivMaps(imgs, probe_layers)
        activ_attrs = activAttrs(activ_maps)
        anno_ids = [[anno[0] for anno in img_annos] for img_annos in annos]
        updateActivAttrDiffs(attr_diffs, activ_attrs, anno_ids, patched=False)
        
        imgs, anno_ids = patch(imgs, annos)
        activ_maps_p = model.getActivMaps(imgs, probe_layers)
        activ_attrs_p = activAttrs(activ_maps_p)
        updateActivAttrDiffs(attr_diffs, activ_attrs_p, anno_ids, patched=True)

    # comparison with identification results
        
