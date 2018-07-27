'''
Inner detectors verifier

To verify the semantic association

'''


import os, sys
import numpy as np
from itertools import product

curr_path = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.join(curr_path, "..")
if root_path not in sys.path:
    sys.path.insert(0, root_path)

from src.config import PATH
    
from utils.helper.data_loader import BatchLoader, getClassName, weightedVal
from utils.helper.data_processor import patch
from utils.helper.file_manager import saveObject, loadObject
from utils.dissection.activ_processor import activAttrs
from utils.model.model_agent import ModelAgent, splitUnitID


'''
Activation attributes processing

'''

def updateActivAttrDiffs(attr_diffs, activ_attrs, anno_ids, patched=False):
    update_idx = 1 if patched else 0
    for unit_id, unit_activ_attrs in activ_attrs.items():
        for img_activ_attrs, img_anno_ids in zip(unit_activ_attrs, anno_ids):
            for aid in img_anno_ids:
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

                    
def computeAttrChange(attr_diffs):
    averages = {}
    for unit_id, unit_diffs in attr_diffs.items():
        layer, _ = splitUnitID(unit_id)
        for aid, adiff in unit_diffs.items():
            adiff = np.asarray(adiff)
            adiff = (adiff[:, 1] / adiff[:, 0]) - 1
            attr_diffs[unit_id][aid] = adiff

            if aid not in averages:
                averages[aid] = {}
            anno_ave = averages[aid]

            if layer not in anno_ave:
                anno_ave[layer] = [[0.0, 0] for i in range(adiff.shape[0])]
            layer_ave = anno_ave[layer]

            for idx, change in enumerate(adiff):
                if not np.isfinite(change):
                    # ignore the cases of NAN and infinite
                    continue
                val_0, count_0 = layer_ave[idx]
                layer_ave[idx][0] = weightedVal(val_0, count_0, change)
                layer_ave[idx][1] += 1

    return averages, attr_diffs


'''
Main program

'''


if __name__ == "__main__":
    bl = BatchLoader(amount=1)
    model = ModelAgent(input_size=1)
    # probe_layers = loadObject(PATH.MODEL.PROBE)
    probe_layers = ["conv5_2", "conv5_3"]
    
    attr_diffs = {}
    while bl:
        batch = bl.nextBatch()
        imgs = batch[1]
        annos = batch[2]

        activ_maps = model.getActivMaps(imgs, probe_layers)
        activ_attrs = activAttrs(activ_maps)
        anno_ids = [[anno[0] for anno in img_annos] for img_annos in annos]
        updateActivAttrDiffs(attr_diffs, activ_attrs, anno_ids, patched=False)
        
        imgs, anno_ids = patch(imgs, annos)
        activ_maps_p = model.getActivMaps(imgs, probe_layers)
        activ_attrs_p = activAttrs(activ_maps_p)
        updateActivAttrDiffs(attr_diffs, activ_attrs_p, anno_ids, patched=True)

    attr_change_aves, attr_changes = computeAttrChange(attr_diffs)
    
    # comparison with identification results
    
