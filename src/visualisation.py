'''
Activation Visualiser

To visualise the feature maps of individual units

'''


import os, sys
from multiprocessing import Pool
from random import randint
from itertools import product

curr_path = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.join(curr_path, "..")
if root_path not in sys.path:
    sys.path.insert(0, root_path)

from src.config import CONFIG, PATH

from utils.helper.data_loader import BatchLoader
from utils.helper.data_mapper import getClassName, getClasses
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

samples = {}
NUM = 1
def visualise(ident, imgs, img_infos, activ_maps, deconvs, num=NUM):
    for ccp, layer, rank, m in nested(ident, depth=3):
        unit = m[0]
        unit_id = unitOfLayer(layer, unit)
        activ_map = activ_maps[unit_id]
        deconv = deconvs[unit_id]
        cls = getClassName(ccp, full=True)
        iou = m[1]

        if ccp not in samples:
            samples[ccp] = {k : set() for k in SAMPLE_TYPES}
        
        for img, info, amap, _deconv in zip(imgs, img_infos, activ_map, deconv):
            img_id = info['id']
            
            smp_type = getSampleType(info, ccp)
            _samples = samples[ccp][smp_type]
            if img_id not in _samples:
                if len(_samples) < num:
                    _samples.add(img_id)
                else:
                    continue
            
            # same image, different units
            img_unit_dir = os.path.join(PATH.OUT.VIS.ROOT,
                                        "{}/images/{}_{}/".format(cls, img_id, smp_type))
            img_unit_path = img_unit_dir + "{}_{}_{}_{:.2f}.png".format(rank, layer, unit, iou)
            vis = revealMask(img, amap, alpha=0.95)
            saveImage(vis, img_unit_path)
            # deconv
            img_unit_path = img_unit_dir + "d_{}_{}_{}_{:.2f}.png".format(rank, layer, unit, iou)
            saveImage(_deconv, img_unit_path)
            # raw image
            img_unit_raw_path = img_unit_dir + "{}.png".format(smp_type)
            if not os.path.exists(img_unit_raw_path):
                saveImage(img, img_unit_raw_path)

            # same unit, different images
            unit_img_dir = os.path.join(PATH.OUT.VIS.ROOT,
                                        "{}/units/{}/{}_{}_{:.2f}/".format(cls, layer, rank, unit, iou))
            unit_img_path = unit_img_dir + "{}_{}.png".format(img_id, smp_type)
            saveImage(vis, unit_img_path)
            unit_img_path = unit_img_dir + "{}_{}_d.png".format(img_id, smp_type)
            saveImage(_deconv, unit_img_path)
            unit_img_raw_path = unit_img_dir + "{}.png".format(img_id)
            if not os.path.exists(unit_img_raw_path):
                saveImage(img, unit_img_raw_path)

            

SAMPLE_TYPES = ['positive', 'negative']
def getSampleType(img_info, ccp):
    img_labels = img_info['classes']
    smp_type = 'positive' if ccp in img_labels else 'negative'
    return smp_type

def finished():
    for ccp, smp_type in product(samples, SAMPLE_TYPES):
        _samples = samples[ccp][smp_type]
        if len(_samples) < NUM:
            return False

    return True

def process(activ_maps, field_maps):
    reflected = reflect(activ_maps, field_maps)
    activ_maps = None
    return reflected

                
'''
Main Program

'''

if __name__ == "__main__":
    batch_size = 5
    bl = BatchLoader(amount=5, batch_size=batch_size, mode='classes')
    model = ModelAgent(input_size=batch_size, deconv=True)
    field_maps = model.getFieldmaps()
    probe_layers = loadObject(PATH.MODEL.PROBE)

    ident = identification(top=10, mode='concept', organise=True)
    # ignore classes other than 0 order
    classes = getClasses()
    ident = {cls : ident[cls] for cls in classes}
    probe_units = poolUnits(ident)

    pool = Pool()
    num = pool._processes
    while bl:
        batch = bl.nextBatch()
        img_ids = batch[0]
        imgs = batch[1]
        img_infos = bl.getImageInfo(img_ids)

        print ("Activation Maps: fetching...")
        activ_maps, switches = model.getActivMaps(imgs, probe_layers)
        activ_maps = {k : activ_maps[k] for k in probe_units}

        # Network Dissection
        reflect_amaps = splitDict(activ_maps, num)
        params = [(amap, field_maps) for amap in reflect_amaps]
        print ("Activation Maps: being projected...")
        reflect_amaps = pool.starmap(process, params)
        print ("Finish reflection")
        reflect_amaps = mergeDict(reflect_amaps)

        # DeConvNet
        deconv_amaps = model.getDeconvMaps(activ_maps, switches)
        
        print ("Visualisation: beginning...")
        visualise(ident, imgs, img_infos, reflect_amaps, deconv_amaps)

        if finished():
            print ("Finish")
            bl.finish()
        
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
#                 unit_img_path = os.path.join(PATH.OUT.VIS, file_name)
#                 quanFilter(deconv, 98)
#                 deconv[deconv>0] = 255
#                 saveImage(deconv, unit_img_path)



        
        
