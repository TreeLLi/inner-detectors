'''
Image Plotter

To provide auxiliary functions for drawing images

'''

import cv2
import numpy as np


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
General-purpose Auxiliary Functions

'''
