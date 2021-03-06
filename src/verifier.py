'''
Inner detectors verifier

To verify the semantic association

'''


import os, sys
import numpy as np
from os.path import join, exists, dirname, abspath
from multiprocessing import Pool

curr_path = dirname(abspath(__file__))
root_path = join(curr_path, "..")
if root_path not in sys.path:
    sys.path.insert(0, root_path)

from src.config import PATH
    
from utils.helper.data_loader import BatchLoader, weightedVal
from utils.helper.data_mapper import getClassName
from utils.helper.plotter import plotFigure
from utils.helper.data_processor import patch
from utils.helper.dstruct_helper import nested, splitList
from utils.helper.file_manager import saveObject, loadObject, saveFigure
from utils.dissection.activ_processor import activAttrs, correlation
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
    data_path = PATH.OUT.UNIT_ATTRS
    
    if not exists(data_path):
        print ("Can not find existing verification data, thus beginning from scratch.")

        input_num = 15
        bl = BatchLoader(batch_size=input_num)
        model = ModelAgent(input_size=input_num)
        probe_layers = loadObject(PATH.MODEL.PROBE)

        attr_diffs = {}
        patch_data = [[], []]
        while bl:
            batch = bl.nextBatch()
            imgs = batch[1]
            annos = batch[2]

            # obtain original activation maps
            print ("Fetching activation maps for specific units ...")
            activ_maps = model.getActivMaps(imgs, probe_layers)
            activ_attrs = activAttrs(activ_maps)
            anno_ids = [[anno[0] for anno in img_annos] for img_annos in annos]
            updateActivAttrDiffs(attr_diffs, activ_attrs, anno_ids, patched=False)

            # split data for multi-processing
            imgs, anno_ids = patch(imgs, annos)
            patch_data[0] += imgs
            patch_data[1] += anno_ids

            while(len(patch_data[0])>=input_num or (len(patch_data[0])>0 and not bl)):
                imgs = patch_data[0][:input_num]
                anno_ids = patch_data[1][:input_num]
                activ_maps_p = model.getActivMaps(imgs, probe_layers)
                activ_attrs_p = activAttrs(activ_maps_p)
                patch_data = [_patch_data[input_num:] for _patch_data in patch_data]
                
                updateActivAttrDiffs(attr_diffs, activ_attrs_p, anno_ids, patched=True)
                
            bl.reportProgress()
            
        attr_change_aves, attr_changes = computeAttrChange(attr_diffs)
        saveObject(attr_changes, data_path)
    else:
        print ("Find existing verification data, beginning analysis.")
        attr_changes = loadObject(data_path)

    # analysis for assessing if identification results correct
    concept_matches = loadObject(PATH.OUT.IDE.DATA.CONCEPT)
    data_x = {}
    data_y = {}
    for ccp, unit, match in nested(concept_matches, depth=2):
        try:
            mean_change = attr_changes[unit][ccp][0]
            if not np.isfinite(mean_change):
                continue
            
            if ccp not in data_x:
                data_x[ccp] = []
                data_y[ccp] = []
                
            data_y[ccp].append(mean_change)
            data_x[ccp].append(match[0])
            
        except:
            continue
            
    # plot
    plot_path = PATH.OUT.VERIFICATION
    for ccp, ccp_x in data_x.items():
        ccp_y = data_y[ccp]
        ccp = getClassName(ccp, full=True)
        params = {'xlim' : (0, 0.6), 'ylim' : (-1, 1)}
        labels = {'x' : 'IoU', 'y' : 'change of average activation'}
        figure = plotFigure(ccp_x, ccp_y, form='spot', labels=labels, params=params)

        coeff, pvalue = correlation(ccp_x, ccp_y)
        text = "coeffi: {:.3f}\npvalue: {:.3f}".format(coeff, pvalue)
        figure.text(0.4, 0.5, text, fontsize=20)
        
        figure_path = join(plot_path, ccp+".png")
        saveFigure(figure, figure_path)
    
