'''
Image Plotter

To provide auxiliary functions for drawing images

'''

import cv2
import numpy as np
from addict import Dict as adict
import matplotlib.pyplot as plt
from matplotlib import rc


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

def revealMask(img, mask, alpha=0.5):
    output = img.copy()

    indices = np.argwhere(mask > 0)
    overlay = np.ones_like(img)
    overlay[indices[:, 0], indices[:, 1]] = img[indices[:, 0], indices[:, 1]]
    # overlay *= 255

    cv2.addWeighted(overlay, alpha, img, 1-alpha, 0, output)
    return output


'''
Figure plot

'''

size = 15
rc('xtick', labelsize=size)
rc('ytick', labelsize=size)
def plotFigure(x, y, title=None, form='line', ticks=None, labels=None, params=None, show=False):
    if title:
        plt.figure(title, figsize=(9, 7))
        plt.title(title, fontsize=size)
    else:
        plt.figure(figsize=(9, 6.5))
        
    plot = _getPlot(form)
    x_type = type(x[0]) if isinstance(x, list) else type(x)
    y_type = type(y[0]) if isinstance(y, list) else type(y)
    if x_type == list and y_type == list:
        for _x, _y in zip(x, y):
            plot(_x, _y)
    elif x_type not in [list, dict] and y_type == list:
        for _y in y:
            plot(x, _y)
    elif x_type not in [list, dict] and y_type == dict:
        for k, _y in y.items():
            plot(x, _y, label=k)
    elif x_type == dict and y_type == dict:
        for k, _y in y.items():
            plot(x[k], _y, label=k)
    elif x_type not in [list, dict] and y_type not in [list, dict]:
        plot(x, y)
    else:
        raise Exception("Error: unknown data types for plotting")

    if params:
        if 'xlim' in params:
            plt.xlim(params['xlim'])
            
        if 'ylim' in params:
            plt.ylim(params['ylim'])

    if ticks:
        if 'x' in ticks:
            plt.xticks(x, ticks['x'])
        if 'y' in ticks:
            plt.yticks(y, ticks['y'])

    if labels:
        if 'x' in labels:
            plt.xlabel(labels['x'], fontsize=size)
        if 'y' in labels:
            plt.ylabel(labels['y'], fontsize=size)
            
    if show:
        plt.show()
    # return plt for further figure saving
    return plt

def _getPlot(form):
    if form == 'line':
        return plt.plot
    elif form == 'spot':
        return plt.scatter
