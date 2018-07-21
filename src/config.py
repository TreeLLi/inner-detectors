'''
Configuration Center

To provide unified configuration over the whole program

'''

import os, sys
from easydict import EasyDict as edict

_curr_path = os.path.dirname(os.path.abspath(__file__))
_root_path = os.path.join(_curr_path, "..")
if _root_path not in sys.path:
    sys.path.insert(0, _root_path)


'''
Config for program parameters

'''


CONFIG = edict()

# datasets
CONFIG.DATA = edict()
CONFIG.DATA.SOURCES = ['PASCAL']
# true, generate descriptive statistics report for datasets
CONFIG.DATA.STATISTICS = True

# dissection
CONFIG.DIS = edict()

# 'fast' or 'normal'
# fast: using stored object files to quickly generate required data
# normal: run through normal procedures, like forward pass and reflection
CONFIG.DIS.MODE = "fast"
CONFIG.DIS.MODEL = "VGG16" # the model to be dissected
#CONFIG.DIS.REFLECT = "interpolation" # "interpolation" or "deconvnet"
CONFIG.DIS.REFLECT = "deconvnet"

CONFIG.DIS.IOU_THRESHOLD = 0.1
CONFIG.DIS.TOP = 3

CONFIG.DIS.REPORT_TEXT = True
CONFIG.DIS.REPORT_FIGURE = False

# models
CONFIG.MODEL = edict()
CONFIG.MODEL.INPUT_DIM = (224, 224 ,3)

    
'''
Config for program paths

'''


PATH = edict()

_data_path = os.path.join(_root_path, "datasets")
_pascal_path = os.path.join(_data_path, "pascal_part")
_output_path = os.path.join(_root_path, "output")
_test_path = os.path.join(_root_path, "test")

# global path
PATH.ROOT = _root_path


# dataset path
PATH.DATA = edict()
PATH.DATA.ROOT = _data_path
PATH.DATA.IMG_MAP = os.path.join(_data_path, "imgs_maps.txt")
PATH.DATA.CLS_MAP = os.path.join(_data_path, "class_maps.txt")
PATH.DATA.IMG_CLS_MAP = os.path.join(_data_path, "img_cls_maps.txt")
PATH.DATA.STATISTICS = os.path.join(_data_path, "annos_statistics.txt")

PATH.DATA.PASCAL = edict({
    "ROOT" : _pascal_path,
    "IMGS" : os.path.join(_pascal_path, "raw_images"),
    "ANNOS" : os.path.join(_pascal_path, "part_annos")
})


# output path
PATH.OUT = edict()
PATH.OUT.ROOT = os.path.join(_root_path, "output/")

if CONFIG.DIS.MODEL == "VGG16":
    _vgg16_path = os.path.join(_output_path, "vgg16")
    PATH.OUT.UNIT_MATCH_REPORT = os.path.join(_vgg16_path, "vgg16_unit_matches.txt")
    PATH.OUT.CONCEPT_MATCH_REPORT = os.path.join(_vgg16_path, "vgg16_concept_matches.txt")
    PATH.OUT.UNIT_MATCHES = os.path.join(_vgg16_path, "vgg16_unit_matches.pkl")
    PATH.OUT.CONCEPT_MATCHES = os.path.join(_vgg16_path, "vgg16_concept_matches.pkl")
    
# model path
_model_path = os.path.join(_root_path, "pre-models")

PATH.MODEL = edict()
PATH.MODEL.ROOT = _model_path
        
if CONFIG.DIS.MODEL == "VGG16":
    PATH.MODEL.PARAM = os.path.join(_model_path, "vgg16/vgg16_param.npy")
    PATH.MODEL.CONFIG = os.path.join(_model_path, "vgg16/vgg16_config.json")
    PATH.MODEL.PROBE = os.path.join(_root_path, "src/vgg16_probe.txt")
    PATH.MODEL.FIELDMAPS = os.path.join(_vgg16_path, "vgg16_fieldmaps.pkl")


# test path
PATH.TEST = edict()
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

def isModeFast():
    return CONFIG.DIS.MODE == 'Fast'
