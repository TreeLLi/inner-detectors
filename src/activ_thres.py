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
from src.indr_matcher import matchActivsAnnos, combineMatches, reportProgress

from utils.helper.data_loader import BatchLoader
from utils.helper.dstruct_helper import splitDict
from utils.helper.file_manager import saveObject, loadObject
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
    bl = BatchLoader(amount=4000, mode="classes")
    probe_layers = loadObject(PATH.MODEL.PROBE)
    model = ModelAgent()
    field_maps = model.getFieldmaps()

    quans = [x for x in range(0, 100, 10)]
    pool = Pool()
    num = pool._processes
    matches = [None for x in range(len(quans))]
    while bl:
        start = time.time()
        batch = bl.nextBatch()
        imgs = batch[1]
        annos = batch[2]

        activ_maps = model.getActivMaps(imgs, probe_layers)
        activ_maps = splitDict(activ_maps, num)
        params = [(amap, field_maps, annos, quans) for amap in activ_maps]
        batch_matches = pool.starmap(process, params)
        print ("Combine matches...")
        for batch_match in batch_matches:
            for idx, bm in enumerate(batch_match):
                matches[idx] = combineMatches(matches[idx], bm)

        reportProgress(start, time.time(), bl.batch_id, len(imgs), bl.size)
    # process results
    path = os.path.join(PATH.OUT.ROOT, "vgg16/activ_thres.pkl")
    saveObject(matches, path)
