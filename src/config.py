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

from utils.model.vgg16 import Vgg16


'''
Config for program parameters

'''


CONFIG = edict()

# dissection
CONFIG.DIS = edict()

CONFIG.DIS.MODEL = "VGG16" # the model to be dissected
CONFIG.DIS.REFLECT = "interpolation" # "interpolation" or "deconvnet"

CONFIG.DIS.IOU_THRESHOLD = 0.3
CONFIG.DIS.TOP = 3

CONFIG.DIS.REPORT_TEXT = True
CONFIG.DIS.REPORT_FIGURE = False

# models
CONFIG.MODEL = edict()
if CONFIG.DIS.MODEL == "VGG16":
    CONFIG.MODEL.ID = "VGG16"
    CONFIG.MODEL.INPUT_DIM = (224, 224, 3)


'''
Config for program paths

'''


PATH = edict()

_data_path = os.path.join(_root_path, "datasets")
_pascal_path = os.path.join(_data_path, "pascal_part")
_output_path = os.path.join(_root_path, "output")


# global path
PATH.ROOT = _root_path


# dataset path
PATH.DATA = edict()
PATH.DATA.ROOT = _data_path

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
    PATH.OUT.MATCH_REPORT = os.path.join(_vgg16_path, "vgg16_matches.txt")
    if not os.path.exists(_vgg16_path):
        os.makedirs(_vgg16_path)
    PATH.OUT.MATCH_OBJECT = os.path.join(_vgg16_path, "vgg16_matches.pkl")

# model path
_model_path = os.path.join(_root_path, "pre-models")

PATH.MODEL = edict()
PATH.MODEL.ROOT = _model_path
        
if CONFIG.DIS.MODEL == "VGG16":
    PATH.MODEL.PARAM = os.path.join(_model_path, "vgg16.npy")
    PATH.MODEL.PROBE = os.path.join(_root_path, "src/probe_vgg16.txt")
    PATH.MODEL.FIELDMAPS = os.path.join(_vgg16_path, "vgg16_fieldmaps.pkl")


'''
Auxilary functions

To determine status

'''

def isVGG16(model):
    if isinstance(model, Vgg16):
        return True
    elif isinstance(model, str) and model.upper()=="VGG16":
        return True
    else:
        return False
