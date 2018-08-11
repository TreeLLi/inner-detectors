'''
Correlation Analysis

To analyse the statistical correlation between unit and unit (unit and classification).

'''


import os, sys
import numpy as np

from itertools import product

curr_path = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.join(curr_path, "..")
if root_path not in sys.path:
    sys.path.insert(0, root_path)

from src.config import CONFIG, PATH

from utils.helper.data_loader import BatchLoader
from utils.helper.file_manager import loadObject, saveObject, saveFigure
from utils.helper.plotter import plotFigure
from utils.helper.dstruct_helper import paired
from utils.model.model_agent import ModelAgent
from utils.dissection.activ_processor import activAttrs, ATTRS, correlation


'''
Activ Attrs Processing

'''

def splitDic(dic, keys):
    split = [{}, {}]
    for key, val in dic.items():
        if any(_key in key for _key in keys):
            split[1][key] = val
        else:
            split[0][key] = val
    return split

def splitAttr(dic):
    split = {}
    for k, v in dic.items():
        attr_num = v.shape[-1]
        for i in range(attr_num):
            if i not in split:
                split[i] = {}
            split[i][k] = v[:, i]
    return split

def integrate(dic, _dic):
    for key, val in _dic.items():
        if key not in dic:
            dic[key] = list(val)
        else:
            dic[key] += list(val)


'''
Match results processing

'''

def organise(matches):
    _matches = {}
    for ccp, unit_id, m in nested(matches):
        if ccp not in _matches:
            _matches[ccp] = {}
        ccp_m = _matches[ccp]

        layer, unit = splitUnitID(unit_id)
        if layer not in ccp_m:
            ccp_m[layer] = {}
        layer_m = ccp_m[layer]
        layer_m[unit] = m[0]

    for ccp, layer, m in nested(_matches):
        _matches[ccp][layer] = sortDict(m)

    return _matches


'''
Repprt & Figures

'''

def reportCorrelations(corrs):
    results = {}
    for unit_1, unit_2, coef in nested(corrs):
        # TODO

'''
Main Program

'''

if __name__ == "__main__":
    data_path = PATH.OUT.COR.ACTIVS
    if os.path.exists(data_path):
        print("Units activaiton data: load from existing file.")
        data = loadObject(data_path)
    else:
        print ("Units activation data: generate from scratch...")
        bl = BatchLoader(sources=["PASCAL"])
        model = ModelAgent()
        probe_layers = loadObject(PATH.MODEL.PROBE)
        cls_layers = model.getLayers()[-2:]
        probe_layers += cls_layers
        
        data = {}
        while bl:
            batch = bl.nextBatch()
            imgs = batch[1]
    
            activ_maps = model.getActivMaps(imgs, probe_layers)
            conv_activs, cls_activs = splitDic(activ_maps, cls_layers)
            conv_attrs = activAttrs(conv_activs)
            
            integrate(data, {**conv_attrs, **cls_activs})
            bl.reportProgress()
        data = {k : np.asarray(v) for k, v in data.items()}
        saveObject(data, data_path)
        
    print ("Correlation: analysis begin")
    #split activation attributes series
    # conv_attrs, cls_attrs = splitDic(data, ["prob", "fc8"])
    # conv_attrs = splitAttr(conv_attrs)
    # for attr_idx, attr_name in zip(conv_attrs, ATTRS):
    #     _attrs = {**conv_attrs[attr_idx], **cls_attrs}
    #     corrs = {}
    #     for unit_1, unit_2 in paired(_attrs):
    #         name = "{}-{}".format(unit_1, unit_2)

    #         x = _attrs[unit_1]
    #         y = _attrs[unit_2]
    #         coef, _ = correlation(x, y)
    #         if unit_1 not in corrs:
    #             corrs[unit_1] = {}
    #         corrs[unit_1][unit_2] = coef
    #         if unit_2 not in corrs:
    #             corrs[unit_2] = {}
    #         corrs[unit_2][unit_1] = coef
            
    #         # draw correlation spot figure
    #         # plt = plotFigure(_attrs[unit_1],
    #         #                  _attrs[unit_2],
    #         #                  title=name,
    #         #                  form="spot")
    #         # file_path = os.path.join(PATH.OUT.COR.FIGURE, name+".png")
    #         # saveFigure(plt, file_path)

    #     reportCorrelations(corrs)

    matches = loadObject(PATH.DATA.CONCEPT_MATCHES)
    matches = organise(matches)
