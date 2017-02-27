import numpy as np

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib import rc
import matplotlib.patches as patches

rc('text', usetex=True)

gs = gridspec.GridSpec(4, 8, width_ratios=[3]*8, height_ratios=[0.5, 3, 3, 3],
                       wspace=0, hspace=0, left=0, right=1, top=1, bottom=0)

fontsize = 25

plt.figure(figsize=(24, 9.5))

for row in range(4):
    for col in range(8):
        if row == 0 and col > 0: continue
        if col == 0:
            plt.subplot(gs[row, col], frameon=False)
            plt.gca().add_patch(patches.Rectangle((0, 0), 1.0, 1.0, color='0.8', alpha=1, zorder=0))
            if row == 1:
                img = plt.imread('pebbles-multiscale.jpg')
                plt.imshow(img, extent=[0.06, 0.94, 0.06, 0.94])
                plt.title(r'\textbf{Multi-scale}', fontsize=fontsize, y=0.98)
            if row == 2:
                img = plt.imread('trees-multiscale.jpg')
                plt.imshow(img, extent=[0.06, 0.94, 0.06, 0.94])
            if row == 3:
                img = plt.imread('brick-multiscale.jpg')
                plt.imshow(img, extent=[0.06, 0.94, 0.06, 0.94])
            plt.xlim([0, 1])
            plt.ylim([0, 1])
            plt.gca().xaxis.set_visible(False)    
            plt.gca().yaxis.set_visible(False)
            
        if col == 1:
            plt.subplot(gs[row, col], frameon=False)
            if row == 1:
                img = plt.imread('pebbles-single-3.jpg')
                plt.imshow(img, extent=[0.06, 0.94, 0.06, 0.94])
                plt.title(r'\textbf{3x3}', fontsize=fontsize, y=0.98)
            if row == 2:
                img = plt.imread('trees-single-3.jpg')
                plt.imshow(img, extent=[0.06, 0.94, 0.06, 0.94])
            if row == 3:
                img = plt.imread('brick-single-3.jpg')
                plt.imshow(img, extent=[0.06, 0.94, 0.06, 0.94])
            plt.xlim([0, 1])
            plt.ylim([0, 1])
            plt.gca().xaxis.set_visible(False)    
            plt.gca().yaxis.set_visible(False)
            
        if col == 2:
            plt.subplot(gs[row, col], frameon=False)
            if row == 1:
                img = plt.imread('pebbles-single-7.jpg')
                plt.imshow(img, extent=[0.06, 0.94, 0.06, 0.94])
                plt.title(r'\textbf{7x7}', fontsize=fontsize, y=0.98)
            if row == 2:
                img = plt.imread('trees-single-7.jpg')
                plt.imshow(img, extent=[0.06, 0.94, 0.06, 0.94])
            if row == 3:
                img = plt.imread('brick-single-7.jpg')
                plt.imshow(img, extent=[0.06, 0.94, 0.06, 0.94])
            plt.xlim([0, 1])
            plt.ylim([0, 1])
            plt.gca().xaxis.set_visible(False)    
            plt.gca().yaxis.set_visible(False)
            
        if col == 3:
            plt.subplot(gs[row, col], frameon=False)
            if row == 1:
                img = plt.imread('pebbles-single-11.jpg')
                plt.imshow(img, extent=[0.06, 0.94, 0.06, 0.94])
                plt.title(r'\textbf{11x11}', fontsize=fontsize, y=0.98)
            if row == 2:
                img = plt.imread('trees-single-11.jpg')
                plt.imshow(img, extent=[0.06, 0.94, 0.06, 0.94])
            if row == 3:
                img = plt.imread('brick-single-11.jpg')
                plt.imshow(img, extent=[0.06, 0.94, 0.06, 0.94])
            plt.xlim([0, 1])
            plt.ylim([0, 1])
            plt.gca().xaxis.set_visible(False)    
            plt.gca().yaxis.set_visible(False)
            
        if col == 4:
            plt.subplot(gs[row, col], frameon=False)
            if row == 1:
                img = plt.imread('pebbles-single-23.jpg')
                plt.imshow(img, extent=[0.06, 0.94, 0.06, 0.94])
                plt.title(r'\textbf{23x23}', fontsize=fontsize, y=0.98)
            if row == 2:
                img = plt.imread('trees-single-23.jpg')
                plt.imshow(img, extent=[0.06, 0.94, 0.06, 0.94])
            if row == 3:
                img = plt.imread('brick-single-23.jpg')
                plt.imshow(img, extent=[0.06, 0.94, 0.06, 0.94])
            plt.xlim([0, 1])
            plt.ylim([0, 1])
            plt.gca().xaxis.set_visible(False)    
            plt.gca().yaxis.set_visible(False)
            
        if col == 5:
            plt.subplot(gs[row, col], frameon=False)
            if row == 1:
                img = plt.imread('pebbles-single-37.jpg')
                plt.imshow(img, extent=[0.06, 0.94, 0.06, 0.94])
                plt.title(r'\textbf{37x37}', fontsize=fontsize, y=0.98)
            if row == 2:
                img = plt.imread('trees-single-37.jpg')
                plt.imshow(img, extent=[0.06, 0.94, 0.06, 0.94])
            if row == 3:
                img = plt.imread('brick-single-37.jpg')
                plt.imshow(img, extent=[0.06, 0.94, 0.06, 0.94])
            plt.xlim([0, 1])
            plt.ylim([0, 1])
            plt.gca().xaxis.set_visible(False)    
            plt.gca().yaxis.set_visible(False)
            
        if col == 6:
            plt.subplot(gs[row, col], frameon=False)
            if row == 1:
                img = plt.imread('pebbles-single-55.jpg')
                plt.imshow(img, extent=[0.06, 0.94, 0.06, 0.94])
                plt.title(r'\textbf{55x55}', fontsize=fontsize, y=0.98)
            if row == 2:
                img = plt.imread('trees-single-55.jpg')
                plt.imshow(img, extent=[0.06, 0.94, 0.06, 0.94])
            if row == 3:
                img = plt.imread('brick-single-55.jpg')
                plt.imshow(img, extent=[0.06, 0.94, 0.06, 0.94])
            plt.xlim([0, 1])
            plt.ylim([0, 1])
            plt.gca().xaxis.set_visible(False)    
            plt.gca().yaxis.set_visible(False)
            
        if col == 7:
            plt.subplot(gs[row, col], frameon=False)
            if row == 1:
                img = plt.imread('pebbles-multi-linear.jpg')
                plt.imshow(img, extent=[0.06, 0.94, 0.06, 0.94])
                plt.title(r"""\textbf{Multi-scale}
\textbf{(linear)}""", fontsize=fontsize, y=0.98)
            if row == 2:
                img = plt.imread('trees-multi-linear.jpg')
                plt.imshow(img, extent=[0.06, 0.94, 0.06, 0.94])
            if row == 3:
                img = plt.imread('brick-multi-linear.jpg')
                plt.imshow(img, extent=[0.06, 0.94, 0.06, 0.94])
            plt.xlim([0, 1])
            plt.ylim([0, 1])
            plt.gca().xaxis.set_visible(False)    
            plt.gca().yaxis.set_visible(False)
            
plt.savefig('figure2.pdf', bbox_inches='tight')
