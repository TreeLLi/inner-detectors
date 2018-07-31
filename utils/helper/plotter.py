'''
Image Plotter

To provide auxiliary functions for drawing images

'''

import cv2
import numpy as np
from addict import Dict as adict
import matplotlib.pyplot as plt


'''
Image mask 

'''

# 5 colors in total
# ratios = np.arange(0.2, 1, 0.2)
# base = np.asarray([255.0, 255.0, 255.0])
# COLORS = [[int(round(b*r)) for b in base] for r in ratios]

COLORS = [[255,0,0], [0,255,0]]

def maskImage(img, masks, alpha=0.5, labels=None):
    output = img.copy()
    
    for idx, mask in enumerate(masks):
        indices = np.argwhere(mask > 0)
        color = COLORS[idx]
        # if two masks overlay, the later one is retained
        overlay = output.copy()
        overlay[indices[:, 0], indices[:, 1]] = color
        cv2.addWeighted(overlay, alpha, output, 1-alpha, 0, output)
        
    return output

'''
Figure plot

'''

def plotFigure(x, y, title=None, form='line', params=None, show=False):
    if title:
        plt.figure(title)
    else:
        plt.figure()
        
    if form == 'line':
        for _x, _y in zip(x, y):
            plt.plot(_x, _y)

    if show:
        plt.show()
    # return plt for further figure saving
    return plt
