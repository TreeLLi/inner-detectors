'''
Activation Threshold Assessor:

To assess the sensitivity of identification results with respect to activation threshold

'''


import os, sys
import time, datetime
from multiprocessing import Pool
import matplotlib

curr_path = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.join(curr_path, "..")
if root_path not in sys.path:
    sys.path.insert(0, root_path)

from src.config import PATH, CONFIG
from src.indr_matcher import matchActivsAnnos, combineMatches

from utils.helper.data_loader import BatchLoader
from utils.helper.dstruct_helper import splitDict, dictMean, sortDict, nested
from utils.helper.file_manager import saveObject, loadObject, saveFigure
from utils.helper.plotter import plotFigure
from utils.model.model_agent import ModelAgent
from utils.dissection.helper import iou
from utils.dissection.activ_processor import reflect


'''
Multiprocessing

'''

def process(activ_maps, field_maps, annos, quans):
    batch_matches = []
    for quan in quans:
        ref_activ_maps = reflect(activ_maps, field_maps, quan=quan)
        batch_match = matchActivsAnnos(ref_activ_maps, annos)
        batch_matches.append(batch_match)
    print ("process finished.")
    return batch_matches

    
'''
Main Program

'''

if __name__ == "__main__":
    quans = [x for x in range(0, 100, 10)]
    file_path = PATH.OUT.ACTIV_THRESH

    if not os.path.exists(file_path):
        print ("Can not find existing match results, thus beginning from scratch.")
        bl = BatchLoader(amount=4000, classes=0)
        probe_layers = loadObject(PATH.MODEL.PROBE)
        model = ModelAgent()
        field_maps = model.getFieldmaps()
        
        matches = [None for x in range(len(quans))]
        while bl:
            batch = bl.nextBatch()
            imgs = batch[1]
            annos = batch[2]

            activ_maps = model.getActivMaps(imgs, probe_layers)
            activ_maps = splitDict(activ_maps, num)
            params = [(amap, field_maps, annos, quans) for amap in activ_maps]
            with Pool() as pool:
                batch_matches = pool.starmap(process, params)
            print ("Combine matches...")
            for batch_match in batch_matches:
                for idx, bm in enumerate(batch_match):
                    matches[idx] = combineMatches(matches[idx], bm)
            bl.reportProgress()
            
        saveObject(matches, file_path)
    else:
        matches = loadObject(file_path)
        print("Find existing match results, thus skipping to analyse results.")

    # match results analysis
    # overall comparison
    # with Pool() as pool:
    #     means = pool.starmap(dictMean, [(m, 0) for m in matches])
    # labels = {'x' : 'quantile', 'y' : 'mean IoU'}
    # plt = plotFigure(quans, means, title="means v.s. quantiles", show=True)
    
    # saveFigure(plt, os.path.join(plot_path, "overall.png"))

    plot_path = os.path.join(PATH.OUT.ROOT, "activ_thres")
    # sort and comparison
    data = {}
    for idx, _matches in enumerate(matches):
        for unit_id, ccp, m in nested(_matches, depth=2):
            if unit_id not in data:
                data[unit_id] = {}
            unit_data = data[unit_id]
                
            if ccp not in unit_data:
                unit_data[ccp] = []
            unit_ccp_data = unit_data[ccp]
            unit_ccp_data.append(m[0])

    labels = {'x' : 'quantile', 'y' : 'IoU'}
    matplotlib.rc('xtick', labelsize=20)
    matplotlib.rc('ytick', labelsize=20)
    for unit_id, ccp_data in data.items():
        file_path = os.path.join(plot_path, unit_id + ".png")
        plt = plotFigure(quans, ccp_data, title=unit_id, labels=labels)
        saveFigure(plt, file_path)
