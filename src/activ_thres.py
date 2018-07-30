'''
Activation Threshold Assessor:

To assess the sensitivity of identification results with respect to activation threshold

'''


import os, sys
import time, datetime
from multiprocessing import Pool

curr_path = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.join(curr_path, "..")
if root_path not in sys.path:
    sys.path.insert(0, root_path)

from src.config import PATH, CONFIG
from src.indr_matcher import matchActivsAnnos, combineMatches

from utils.helper.data_loader import BatchLoader
from utils.helper.file_manager import saveObject, loadObject
from utils.model.model_agent import ModelAgent
from utils.dissection.helper import iou
from utils.dissection.activ_processor import reflect


'''
Multiprocessing

'''

def process(activ_maps, field_maps, annos, quan, matches, idx):
    print ("Processing Quantile {}%".format(quan))
    ref_activ_maps = reflect(activ_maps, field_maps, quan=quan)
    batch_matches = matchActivsAnnos(ref_activ_maps, annos)
    matches[idx] = combineMatches(matches[idx], batch_matches)
    print ("Finished processing Quantile {}%".format(quan))

    
'''
Main Program

'''

if __name__ == "__main__":
    bl = BatchLoader(amount=20, mode="classes")
    probe_layers = loadObject(PATH.MODEL.PROBE)
    model = ModelAgent()
    field_maps = model.getFieldmaps()

    quans = [x for x in range(0, 100, 10)]
    pool = Pool()
    num = pool._processes
    matches = [None for x in range(len(quans))]
    while bl:
        batch = bl.nextBatch()
        imgs = batch[1]
        annos = batch[2]

        activ_maps = model.getActivMaps(imgs, probe_layers)
        params = []
        for idx, quan in enumerate(quans):
            params.append((activ_maps, field_maps, annos, quan, matches, idx))
        pool.starmap(process, params)
    # process results
