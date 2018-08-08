'''
Correlation Analysis

To analyse the statistical correlation between unit and unit (unit and classification).

'''


import os, sys
import numpy as np

curr_path = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.join(curr_path, "..")
if root_path not in sys.path:
    sys.path.insert(0, root_path)

from src.config import CONFIG, PATH

from utils.helper.data_loader import BatchLoader
from utils.helper.file_manager import loadObject, saveObject
from utils.helper.model_agent import ModelAgent

from utils.dissection.activ_processor import activAttrs


'''
Activ Attrs Processing

'''

def split(dic, keys)

def integrate(data, values):
    


'''
Main Program

'''

if __name__ == "__main__":
    data_path = PATH.OUT.UNIT_ACTIVS
    if os.path.exists(data_path):
        print("Units activaiton data: load from existing file.")
        data = loadObject(data_path)
    else:
        print ("Units activation data: generate from scratch...")
        bl = BatchLoader(amount=10)
        model = ModelAgent()
        probe_layers = loadObject(PATH.MODEL.PROBE)
        cls_layers = model.getLayers()[-2:]
        probe_layers += cls_layers
        
        data = {}
        while bl:
            batch = bl.nextBatch()
            imgs = batch[1]

            activ_maps = model.getActivMaps(imgs, probe_layers)
            conv_activs, cls_activs = split(activ_maps, cls_layers)
            conv_attrs = activAttrs(conv_activs)
            
            integrate(data, {**conv_attrs, **cls_activs})

        data = {k : np.asarray(v) for k, v in data.items()}
        saveObject(data, data_path)
        
    print ("Correlation: analysis begin")
    
