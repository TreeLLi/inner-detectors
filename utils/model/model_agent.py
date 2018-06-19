'''
Model agent:

To provide a single unified interface to invoke models

'''

import os, sys
import tensorflow as tf

curr_path = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.join(curr_path, "../..")
if root_path not in sys.path:
    sys.path.append(root_path)

from vgg16 import Vgg16

from src.config import PATH


class ModelAgent:

    def __init__(self, model="vgg16"):
        # initialise base model according to the given str
        if model.lower() == "vgg16":
            self.model = Vgg16(PATH.MODEL.VGG16.PARAM)

    def getActivMaps(self, imgs):
        if isinstance(self.model, Vgg16):
            # base model VGG16
            inp = tf.placeholder("float", imgs.shape)
            feed_dict = {inp : imgs}
            self.model.build(inp)

            with tf.Session() as sess:
                sess.run(self.model.prob, feed_dict=feed_dict)

            return _getActivMaps(self.model)



def _getActivMaps(model):
    if isinstance(model, Vgg16):
        layers = loadListFromText(PATH.MODEL.VGG16.LAYERS)
        
