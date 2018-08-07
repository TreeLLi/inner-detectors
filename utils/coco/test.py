from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab

pylab.rcParams['figure.figsize'] = (8.0, 10.0)

dataDir = '../..'
dataType = 'val2017'
annFile = "{}/annotations/instances_{}.json".format(dataDir, dataType)

coco = COCO(annFile)
catIds = coco.getCatIds(catNms=['person'])
imgIds = coco.getImgIds(catIds=catIds)
img = coco.loadImgs(imgIds)[0]
print (img)

annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
anns = coco.loadAnns(annIds)
anns = np.asarray(anns)
print (anns.shape, anns)
