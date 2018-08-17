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
from utils.helper.data_mapper import getClassName
from utils.helper.dstruct_helper import nested, splitDict, mergeDict
from utils.helper.file_manager import saveObject, loadObject, saveImage
from utils.helper.plotter import revealMask
from utils.dissection.helper import quanFilter
from utils.dissection.activ_processor import reflect
from utils.dissection.identification import loadIdent, conceptsOfUnit
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

SAMPLES = {}
NUM = 1
def visualise(ident, imgs, img_infos, activ_maps=None, deconvs=None):
    if activ_maps is None and deconvs is None:
        raise Exception("Error - visualise: no data for visualisation.")
    
    global SAMPLES
    for ccp, layer, rank, m in nested(ident, depth=3):
        unit = m[0]
        iou = m[1]
        unit_id = unitOfLayer(layer, unit)
        cls = getClassName(ccp, full=True)

        if activ_maps:
            activ_map = activ_maps[unit_id]
        if deconvs:
            deconv = deconvs[unit_id]

        # create samples counter
        if ccp not in SAMPLES:
            SAMPLES[ccp] = {k : set() for k in SAMPLE_TYPES}
        if unit_id not in SAMPLES:
            SAMPLES[unit_id] = {k : set() for k in SAMPLE_TYPES}
            
        for idx, img in enumerate(imgs):
            info = img_infos[idx]
            img_id = info['id']
            ccp_type, unit_type = getSampleType(img_id, info, ccp, unit_id)

            ccp_samples = SAMPLES[ccp][ccp_type]
            if img_id in ccp_samples:
                save_ccp_sample = True
            elif len(ccp_samples) < NUM:
                save_ccp_sample = True
                ccp_samples.add(img_id)
            else:
                save_ccp_sample = False
            if save_ccp_sample:
                # same image, different units
                img_unit_dir = os.path.join(PATH.OUT.VIS.ROOT, "{}/images/{}_{}/"
                                            .format(cls, img_id, ccp_type))
                if activ_maps:
                    amap = activ_map[idx]
                    vis = revealMask(img, amap, alpha=0.95)
                    img_unit_path = img_unit_dir
                    img_unit_path += "{}_{}_{}_{:.2f}_{}.png".format(rank, layer, unit, iou, unit_type)
                    saveImage(vis, img_unit_path)
                if deconvs:
                    _deconv = deconv[idx]
                    img_unit_path = img_unit_dir
                    img_unit_path += "d_{}_{}_{}_{:.2f}_{}.png".format(rank, layer, unit, iou, unit_type)
                    saveImage(_deconv, img_unit_path)
                # raw image
                img_unit_raw_path = img_unit_dir + "{}.png".format(ccp_type)
                if not os.path.exists(img_unit_raw_path):
                    saveImage(img, img_unit_raw_path)

            unit_samples = SAMPLES[unit_id][unit_type]
            if img_id in unit_samples:
                save_unit_sample = True
            elif len(unit_samples) < NUM:
                save_unit_sample = True
                unit_samples.add(img_id)
            else:
                save_unit_sample = False
            if save_unit_sample:
                # same unit, different images
                unit_img_dir = os.path.join(PATH.OUT.VIS.ROOT, "{}/units/{}/{}_{}_{:.2f}/"
                                            .format(cls, layer, rank, unit, iou))
                if activ_maps:
                    amap = activ_map[idx]
                    vis = revealMask(img, amap, alpha=0.95)
                    unit_img_path = unit_img_dir + "{}_{}.png".format(img_id, unit_type)
                    saveImage(vis, unit_img_path)
                if deconvs:
                    _deconv = deconv[idx]
                    unit_img_path = unit_img_dir + "{}_{}_d.png".format(img_id, unit_type)
                    saveImage(_deconv, unit_img_path)
                # raw image
                unit_img_raw_path = unit_img_dir + "{}.png".format(img_id)
                if not os.path.exists(unit_img_raw_path):
                    saveImage(img, unit_img_raw_path)

SAMPLE_TYPES = ['positive', 'negative']
def getSampleType(img_id, img_info, ccp, unit_id):
    smp_types = [None, None]
    # type of sample for the concept
    ccp_samples = SAMPLES[ccp]
    for smp_type, _samples in ccp_samples.items():
        if img_id in _samples:
            smp_types[0] = smp_type
    img_labels = img_info['classes']
    if not smp_types[0]:
        smp_types[0] = 'positive' if ccp in img_labels else 'negative'
        
    # type of samples for the unit
    unit_samples = SAMPLES[unit_id]
    for smp_type, _samples in unit_samples.items():
        if img_id in _samples:
            smp_types[1] = smp_type
    if not smp_types[1]:
        unit_ccps = conceptsOfUnit(unit_id, top=10)
        _type = any(label in unit_ccps for label in img_labels)
        _type = 'positive' if _type else 'negative'
        smp_types[1] = _type
    return smp_types
    
def finished():
    for ccp, smp_type in product(SAMPLES, SAMPLE_TYPES):
        samples = SAMPLES[ccp][smp_type]
        if len(samples) < NUM:
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

    ident = loadIdent(top=10, mode='concept', organise=True, filtering=0)
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
        # deconv_amaps = model.getDeconvMaps(activ_maps, switches)
        
        print ("Visualisation: beginning...")
        visualise(ident, imgs, img_infos, activ_maps=reflect_amaps) # , deconvs=deconv_amaps)

        if finished():
            print ("Finish")
            bl.finish()
        
        bl.reportProgress()
