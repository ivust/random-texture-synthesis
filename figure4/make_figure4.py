import numpy as np
import os
import ..vgg_loss
import evaluate_quality as eq

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib import rc
import matplotlib.patches as patches

rc('text', usetex=True)

def make_10_notation(x):
    x2 = x / 1e-3
    x2str = r'${:.3f}'.format(x2)
    return x2str + r' \cdot 10^{-3}$'

texture_files1 = ['lena.jpg', 'barbara.jpg', 'trees.jpg', 'pebbles.jpg', 'radish.jpg', 'brick_wall.jpg', 'bubbly_0067.jpg', 'bubbly_0111.jpg', 'bumpy_0121.jpg', 'cobwebbed_0132.jpg']
texture_files2 = ['frilly_0103.jpg', 'gauzy_0126.jpg', 'grid_0053.jpg', 'lacelike_0004.jpg', 'lacelike_0069.jpg', 'lacelike_0073.jpg', 'meshed_0164.jpg', 'pebble_texture.jpg', 'porous_0137.jpg', 'scaly_0127.jpg']

width_ratios  = [3,3,3,3,2,3,3,3,3]
height_ratios = [1.0, 3] + [0.4, 3]*9
gs = gridspec.GridSpec(20, 9, width_ratios=width_ratios, height_ratios=height_ratios,
                       wspace=0, hspace=0, left=0, right=1, top=1, bottom=0)

fontsize = 25

plt.figure(figsize=(sum(width_ratios), sum(height_ratios)))

for row in range(20):
    for col in range(9):
        if col == 0:
            plt.subplot(gs[row, col], frameon=False)
            plt.gca().add_patch(patches.Rectangle((0, 0), 1.0, 1.0, color='0.8', alpha=1, zorder=0))
            if row > 0 and row % 2 == 1:
                img = plt.imread(os.path.join('textures', texture_files1[row//2]))
                plt.imshow(img, extent=[0.06, 0.94, 0.06, 0.94])
                if row == 1:
                    plt.title(r'\textbf{Original}', fontsize=fontsize, y=1.1)
            plt.xlim([0, 1])
            plt.ylim([0, 1])
            plt.gca().xaxis.set_visible(False)    
            plt.gca().yaxis.set_visible(False)

        if col == 1:
            plt.subplot(gs[row, col], frameon=False)
            if row > 0 and row % 2 == 1:
                img = plt.imread(os.path.join('textures', texture_files1[row//2] + '-11.jpg'))
                plt.imshow(img, extent=[0.06, 0.94, 0.06, 0.94])
                loss = vgg_loss.main(os.path.join('textures', texture_files1[row//2]),
                                     os.path.join('textures', texture_files1[row//2] + '-11.jpg'))
                if row == 1:
                    title = r'\textbf{signle-scale}' + '\n' + make_10_notation(loss)
                else:
                    title = make_10_notation(loss)
                plt.title(title, fontsize=fontsize, y=0.98)
            plt.xlim([0, 1])
            plt.ylim([0, 1])
            plt.gca().xaxis.set_visible(False)    
            plt.gca().yaxis.set_visible(False)

        if col == 2:
            plt.subplot(gs[row, col], frameon=False)
            if row > 0 and row % 2 == 1:
                img = plt.imread(os.path.join('textures', texture_files1[row//2] + '-multiscale.jpg'))
                plt.imshow(img, extent=[0.06, 0.94, 0.06, 0.94])
                loss = vgg_loss.main(os.path.join('textures', texture_files1[row//2]),
                                     os.path.join('textures', texture_files1[row//2] + '-multiscale.jpg'))
                if row == 1:
                    title = r'\textbf{multi-scale}' + '\n' + make_10_notation(loss)
                else:
                    title = make_10_notation(loss)
                plt.title(title, fontsize=fontsize, y=0.98)
            plt.xlim([0, 1])
            plt.ylim([0, 1])
            plt.gca().xaxis.set_visible(False)    
            plt.gca().yaxis.set_visible(False)

        if col == 3:
            plt.subplot(gs[row, col], frameon=False)
            if row > 0 and row % 2 == 1:
                img = plt.imread(os.path.join('textures', texture_files1[row//2] + '-vgg19.jpg'))
                plt.imshow(img, extent=[0.06, 0.94, 0.06, 0.94])
                loss = vgg_loss.main(os.path.join('textures', texture_files1[row//2]),
                                     os.path.join('textures', texture_files1[row//2] + '-vgg19.jpg'))
                if row == 1:
                    title = r'\textbf{Gatys et al. [1]}' + '\n' + make_10_notation(loss)
                else:
                    title = make_10_notation(loss)
                plt.title(title, fontsize=fontsize, y=0.98)
            plt.xlim([0, 1])
            plt.ylim([0, 1])
            plt.gca().xaxis.set_visible(False)    
            plt.gca().yaxis.set_visible(False)
            
        if col == 5:
            plt.subplot(gs[row, col], frameon=False)
            plt.gca().add_patch(patches.Rectangle((0, 0), 1.0, 1.0, color='0.8', alpha=1, zorder=0))
            if row > 0 and row % 2 == 1:
                img = plt.imread(os.path.join('textures', texture_files2[row//2]))
                plt.imshow(img, extent=[0.06, 0.94, 0.06, 0.94])
                if row == 1:
                    plt.title(r'\textbf{Original}', fontsize=fontsize, y=1.1)
            plt.xlim([0, 1])
            plt.ylim([0, 1])
            plt.gca().xaxis.set_visible(False)    
            plt.gca().yaxis.set_visible(False)

        if col == 6:
            plt.subplot(gs[row, col], frameon=False)
            if row > 0 and row % 2 == 1:
                img = plt.imread(os.path.join('textures', texture_files2[row//2] + '-11.jpg'))
                plt.imshow(img, extent=[0.06, 0.94, 0.06, 0.94])
                loss = vgg_loss.main(os.path.join('textures', texture_files2[row//2]),
                                     os.path.join('textures', texture_files2[row//2] + '-11.jpg'))
                if row == 1:
                    title = r'\textbf{signle-scale}' + '\n' + make_10_notation(loss)
                else:
                    title = make_10_notation(loss)
                plt.title(title, fontsize=fontsize, y=0.98)
            plt.xlim([0, 1])
            plt.ylim([0, 1])
            plt.gca().xaxis.set_visible(False)    
            plt.gca().yaxis.set_visible(False)

        if col == 7:
            plt.subplot(gs[row, col], frameon=False)
            if row > 0 and row % 2 == 1:
                img = plt.imread(os.path.join('textures', texture_files2[row//2] + '-multiscale.jpg'))
                plt.imshow(img, extent=[0.06, 0.94, 0.06, 0.94])
                loss = vgg_loss.main(os.path.join('textures', texture_files2[row//2]),
                                     os.path.join('textures', texture_files2[row//2] + '-multiscale.jpg'))
                if row == 1:
                    title = r'\textbf{multi-scale}' + '\n' + make_10_notation(loss)
                else:
                    title = make_10_notation(loss)
                plt.title(title, fontsize=fontsize, y=0.98)
            plt.xlim([0, 1])
            plt.ylim([0, 1])
            plt.gca().xaxis.set_visible(False)    
            plt.gca().yaxis.set_visible(False)

        if col == 8:
            plt.subplot(gs[row, col], frameon=False)
            if row > 0 and row % 2 == 1:
                img = plt.imread(os.path.join('textures', texture_files2[row//2] + '-vgg19.jpg'))
                plt.imshow(img, extent=[0.06, 0.94, 0.06, 0.94])
                loss = vgg_loss.main(os.path.join('textures', texture_files2[row//2]),
                                     os.path.join('textures', texture_files2[row//2] + '-vgg19.jpg'))
                if row == 1:
                    title = r'\textbf{Gatys et al. [1]}' + '\n' + make_10_notation(loss)
                else:
                    title = make_10_notation(loss)
                plt.title(title, fontsize=fontsize, y=0.98)
            plt.xlim([0, 1])
            plt.ylim([0, 1])
            plt.gca().xaxis.set_visible(False)    
            plt.gca().yaxis.set_visible(False)

plt.savefig('figure4.pdf', bbox_inches='tight')
