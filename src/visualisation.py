'''
Activation Visualiser

To visualise the feature maps of individual units

'''


import os, sys
from multiprocessing import Pool
from contextlib import closing

curr_path = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.join(curr_path, "..")
if root_path not in sys.path:
    sys.path.insert(0, root_path)

from src.config import CONFIG, PATH

from utils.helper.data_loader import BatchLoader
from utils.helper.data_mapper import getClassName
from utils.helper.dstruct_helper import nested, splitDict, mergeDict
from utils.helper.file_manager import saveObject, loadObject, saveImage
from utils.helper.plotter import revealMask
from utils.dissection.helper import quanFilter
from utils.dissection.activ_processor import reflect
from utils.dissection.identification import identification
from utils.model.model_agent import ModelAgent, unitOfLayer


'''
Auxiliary

'''

def poolUnits(ident):
    pool = set()
    for ccp, layer, rank, m in nested(ident, depth=3):
        unit = m[0]
        unit_id = unitOfLayer(layer, unit)
        pool.add(unit_id)
    return pool


'''
Visualisation

'''

def visualise(ident, imgs, img_infos, activ_maps):
    for ccp, layer, rank, m in nested(ident, depth=3):
        unit = m[0]
        unit_id = unitOfLayer(layer, unit)
        activ_map = activ_maps[unit_id]
        cls = getClassName(ccp, full=True)
        
        for img, info, amap in zip(imgs, img_infos, activ_map):
            img_id = info['id']
            img_labels = info['classes']

            sample = 'positive' if ccp in img_labels else 'negative'
            file_path = "{}/{}/{}_{}/{}_{}_{:.2f}.png".format(cls,
                                                              layer,
                                                              img_id,
                                                              sample,
                                                              rank,
                                                              unit,
                                                              m[1])
            file_path = os.path.join(PATH.OUT.VIS.ROOT, file_path)
            
            vis = revealMask(img, amap, alpha=0.85)
            saveImage(vis, file_path)

            file_path = "{}/{}/{}_{}/{}.png".format(cls, layer, img_id, sample, sample)
            file_path = os.path.join(PATH.OUT.VIS.ROOT, file_path)
            if not os.path.exists(file_path):
                saveImage(img, file_path)
            

def process(activ_maps, field_maps):
    print (len(activ_maps))
    reflected = reflect(activ_maps, field_maps)
    activ_maps = None
    print ("Finish")
    return reflected

                
'''
Main Program

'''

if __name__ == "__main__":
    test = 10
    bl = BatchLoader(amount=test)
    model = ModelAgent(input_size=test)
    field_maps = model.getFieldmaps()
    probe_layers = loadObject(PATH.MODEL.PROBE)

    # ident = identification(top=10, mode='concept', organise=True)
    # probe_units = poolUnits(ident)
    
    num = 8
    while bl:
        batch = bl.nextBatch()
        img_ids = batch[0]
        imgs = batch[1]
        img_infos = bl.getImageInfo(img_ids)

        print ("Activation Maps: fetching...")
        activ_maps = model.getActivMaps(imgs, probe_layers)
        #activ_maps = {k : activ_maps[k] for k in probe_units}
        activ_maps = splitDict(activ_maps, num)
        params = [(amap, field_maps) for amap in activ_maps]
        print ("Activation Maps: being projected...")
        with closing(Pool()) as pool:
            reflected = pool.starmap(process, params)
        activ_maps = None
        #reflected = mergeDict(reflected)
        print ("Finish reflection")
        merged = {}
        for amap in reflected:
            print ("Merging")
            merged = {**merged, **amap}
        print (len(merged))
        # reflected = reflect(reflected, field_maps)
        
        print ("Visualisation: beginning...")
        #visualise(ident, imgs, img_infos, reflected)

        bl.reportProgress()












        
# if __name__ == "__main__":
#     test = 10
#     bl = BatchLoader(amount=test)
#     model = ModelAgent(input_size=test, deconv=True)
#     #probe_layers = loadObject(PATH.MODEL.PROBE)
#     probe_layers = ["conv5_2"]
    
#     while bl:
#         batch = bl.nextBatch()
#         ids = batch[0]
#         imgs = batch[1]

#         activ_maps, switches = model.getActivMaps(imgs, probe_layers)
#         activ_maps = {k : activ_maps[k] for k in ["conv5_2_0", "conv5_2_200"]}
#         for k, v in activ_maps.items():
#             quanFilter(v, 80, sequence=True)
#         deconvs = model.getDeconvMaps(activ_maps, switches)

#         for unit, _deconvs in deconvs.items():
#             for img_idx, deconv in enumerate(_deconvs):
#                 img_id = ids[img_idx]
#                 labels = bl.getImgLabels(img_id)
#                 cls = getClassName(labels[0], full=True)
#                 file_name = "{}/{}.png".format(unit, cls)
#                 file_path = os.path.join(PATH.OUT.VIS, file_name)
#                 quanFilter(deconv, 98)
#                 deconv[deconv>0] = 255
#                 saveImage(deconv, file_path)



        
        
