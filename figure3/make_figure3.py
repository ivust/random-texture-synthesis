import numpy as np

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib import rc
import matplotlib.patches as patches

rc('text', usetex=True)

gs = gridspec.GridSpec(4, 4, width_ratios=[3]*4, height_ratios=[0.5, 3, 3, 3],
                       wspace=0, hspace=0, left=0, right=1, top=1, bottom=0)

fontsize = 25

plt.figure(figsize=(12, 9.5))

for row in range(4):
    for col in range(4):
        if row == 0 and col > 0: continue
        if col == 0:
            plt.subplot(gs[row, col], frameon=False)
            plt.gca().add_patch(patches.Rectangle((0, 0), 1.0, 1.0, color='0.8', alpha=1, zorder=0))
            if row == 1:
                img = plt.imread('pebbles.jpg')
                plt.imshow(img, extent=[0.06, 0.94, 0.06, 0.94])
                plt.title(r'\textbf{Original}', fontsize=fontsize, y=0.98)
            if row == 2:
                img = plt.imread('trees.jpg')
                plt.imshow(img, extent=[0.06, 0.94, 0.06, 0.94])
            if row == 3:
                img = plt.imread('brick_wall.jpg')
                plt.imshow(img, extent=[0.06, 0.94, 0.06, 0.94])
            plt.xlim([0, 1])
            plt.ylim([0, 1])
            plt.gca().xaxis.set_visible(False)    
            plt.gca().yaxis.set_visible(False)
            
        if col == 1:
            plt.subplot(gs[row, col], frameon=False)
            if row == 1:
                img = plt.imread('pebbles-sample-1.jpg')
                plt.imshow(img, extent=[0.06, 0.94, 0.06, 0.94])
                plt.title(r'\textbf{Sample 1}', fontsize=fontsize, y=0.98)
            if row == 2:
                img = plt.imread('trees-sample-1.jpg')
                plt.imshow(img, extent=[0.06, 0.94, 0.06, 0.94])
            if row == 3:
                img = plt.imread('brick-sample-1.jpg')
                plt.imshow(img, extent=[0.06, 0.94, 0.06, 0.94])
            plt.xlim([0, 1])
            plt.ylim([0, 1])
            plt.gca().xaxis.set_visible(False)    
            plt.gca().yaxis.set_visible(False)
            
        if col == 2:
            plt.subplot(gs[row, col], frameon=False)
            if row == 1:
                img = plt.imread('pebbles-sample-2.jpg')
                plt.imshow(img, extent=[0.06, 0.94, 0.06, 0.94])
                plt.title(r'\textbf{Sample 2}', fontsize=fontsize, y=0.98)
            if row == 2:
                img = plt.imread('trees-sample-2.jpg')
                plt.imshow(img, extent=[0.06, 0.94, 0.06, 0.94])
            if row == 3:
                img = plt.imread('brick-sample-2.jpg')
                plt.imshow(img, extent=[0.06, 0.94, 0.06, 0.94])
            plt.xlim([0, 1])
            plt.ylim([0, 1])
            plt.gca().xaxis.set_visible(False)    
            plt.gca().yaxis.set_visible(False)
            
        if col == 3:
            plt.subplot(gs[row, col], frameon=False)
            if row == 1:
                img = plt.imread('pebbles-sample-3.jpg')
                plt.imshow(img, extent=[0.06, 0.94, 0.06, 0.94])
                plt.title(r'\textbf{Sample 3}', fontsize=fontsize, y=0.98)
            if row == 2:
                img = plt.imread('trees-sample-3.jpg')
                plt.imshow(img, extent=[0.06, 0.94, 0.06, 0.94])
            if row == 3:
                img = plt.imread('brick-sample-3.jpg')
                plt.imshow(img, extent=[0.06, 0.94, 0.06, 0.94])
            plt.xlim([0, 1])
            plt.ylim([0, 1])
            plt.gca().xaxis.set_visible(False)    
            plt.gca().yaxis.set_visible(False)
            
plt.savefig('figure3.pdf', bbox_inches='tight')
