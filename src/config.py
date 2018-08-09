'''
Configuration Center

To provide unified configuration over the whole program

'''

import sys
from os.path import join, dirname, abspath
from addict import Dict as adict

_curr_path = dirname(abspath(__file__))
_root_path = join(_curr_path, "..")
if _root_path not in sys.path:
    sys.path.insert(0, _root_path)


'''
Config for program parameters

'''


CONFIG = adict()

# datasets
CONFIG.DATA.SOURCES = ['PASCAL', 'COCO']
# true, generate descriptive statistics report for datasets
CONFIG.DATA.STATISTICS = True

# dissection

# 'fast' or 'normal'
# fast: using stored object files to quickly generate required data
# normal: run through normal procedures, like forward pass and reflection
CONFIG.DIS.MODE = "fast"
CONFIG.DIS.MODEL = "VGG16" # the model to be dissected
CONFIG.DIS.REFLECT = "interpolation" # "interpolation" or "deconvnet"

CONFIG.DIS.IOU_THRESHOLD = 0.1
CONFIG.DIS.TOP = 3
CONFIG.DIS.QUANTILE = 50
CONFIG.DIS.ANNO_PIXEL_THRESHOLD = 0

CONFIG.DIS.REPORT_TEXT = True
CONFIG.DIS.REPORT_FIGURE = False

# models
CONFIG.MODEL.INPUT_DIM = (224, 224, 3)

    
'''
Config for program paths

'''


PATH = adict()

_data_path = join(_root_path, "datasets")
_output_path = join(_root_path, "output")
_test_path = join(_root_path, "test")

# global path
PATH.ROOT = _root_path


# dataset path
PATH.DATA.ROOT = _data_path
PATH.DATA.IMG_MAP = join(_data_path, "img_ids.txt")
PATH.DATA.CLS_MAP = join(_data_path, "class_maps.txt")
PATH.DATA.IMG_CLS_MAP = join(_data_path, "img_cls_maps.txt")
PATH.DATA.STATISTICS = join(_data_path, "annos_statistics.txt")

_pascal_path = join(_data_path, "pascal_part")
PATH.DATA.PASCAL = adict({
    "ROOT" : _pascal_path,
    "IMGS" : join(_pascal_path, "raw_images"),
    "ANNOS" : join(_pascal_path, "part_annos")
})

_coco_path = join(_data_path, "ms_coco")
PATH.DATA.COCO = adict({
    "ROOT" : _coco_path,
    "ANNOS" : join(_coco_path, "annotations/instances_{}2017.json"),
    "IMGS" : join(_coco_path, "images/{}2017")
})


# output path
PATH.OUT.ROOT = join(_root_path, "output/")

if CONFIG.DIS.MODEL == "VGG16":
    _vgg16_path = join(_output_path, "vgg16")
    PATH.OUT.UNIT_MATCH_REPORT = join(_vgg16_path, "vgg16_unit_matches.txt")
    PATH.OUT.CONCEPT_MATCH_REPORT = join(_vgg16_path, "vgg16_concept_matches.txt")
    PATH.OUT.UNIT_MATCHES = join(_vgg16_path, "vgg16_unit_matches.pkl")
    PATH.OUT.CONCEPT_MATCHES = join(_vgg16_path, "vgg16_concept_matches.pkl")

    PATH.OUT.ACTIV_THRESH = join(_vgg16_path, "activ_thres.pkl")

    PATH.OUT.UNIT_ATTRS = join(_vgg16_path, "unit_attrs.pkl")
    PATH.OUT.VERIFICATION = join(_vgg16_path, "verification")

    # correlation results
    PATH.OUT.COR.ROOT = join(_vgg16_path, "correlation")
    PATH.OUT.COR.ACTIVS = join(PATH.OUT.COR.ROOT, "unit_activations.pkl")
    PATH.OUT.COR.FIGURE = join(PATH.OUT.COR.ROOT, "figures")
    PATH.OUT.COR.REPORT = join(PATH.OUT.COR.ROOT, "correlation_report.txt")
    
    
# model path
_model_path = join(_root_path, "pre-models")

PATH.MODEL.ROOT = _model_path        
if CONFIG.DIS.MODEL == "VGG16":
    PATH.MODEL.PARAM = join(_model_path, "vgg16/vgg16_param.npy")
    PATH.MODEL.CONFIG = join(_model_path, "vgg16/vgg16_config.json")
    PATH.MODEL.PROBE = join(_root_path, "src/vgg16_probe.txt")
    PATH.MODEL.FIELDMAPS = join(_vgg16_path, "vgg16_fieldmaps.pkl")


# test path
PATH.TEST.ROOT = _test_path
    
'''
Auxilary functions

To determine status

'''


def isPASCAL(source):
    if isinstance(source, str) and source.upper()=='PASCAL':
        return True
    else:
        return False

def isCOCO(source):
    if isinstance(source, str) and source.upper()=='COCO':
        return True
    else:
        return False
