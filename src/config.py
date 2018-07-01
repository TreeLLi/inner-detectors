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

# models
CONFIG.MODEL = edict()

CONFIG.MODEL.RANGE = ["vgg16"]

CONFIG.MODEL.VGG16 = edict()
CONFIG.MODEL.VGG16.ID = "vgg16"
CONFIG.MODEL.VGG16.INPUT_DIM = (224, 224, 3)

# dissection
CONFIG.DIS = edict()

CONFIG.DIS.MODEL = CONFIG.MODEL.VGG16.ID # the model to be dissected
CONFIG.DIS.REFLECT = "linear" # "linear" or "deconvnet"

CONFIG.DIS.IOU_THRESHOLD = 0.0
CONFIG.DIS.TOP = 1

CONFIG.DIS.REPORT_TEXT = True
CONFIG.DIS.REPORT_FIGURE = False


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


# model path
_model_path = os.path.join(_root_path, "pre-models")
_vgg16_path = os.path.join(_model_path, "vgg16.npy")

PATH.MODEL = edict()
PATH.MODEL.ROOT = _model_path

# VGG16
PATH.MODEL.VGG16 = edict()
PATH.MODEL.VGG16.PARAM = _vgg16_path
PATH.MODEL.VGG16.LAYERS = os.path.join(_root_path, "src/layers_vgg16.txt")


# output path
PATH.OUT = edict()
PATH.OUT.MATCH = os.path.join(_root_path, "output/")


'''
Auxilary functions

To determine status

'''

def isVGG16(model):
    if isinstance(model, Vgg16):
        return True
    elif isinstance(model, str) and model.lower()==CONFIG.MODEL.VGG16.ID:
        return True
    else:
        return False
