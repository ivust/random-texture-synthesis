import numpy as np
import scipy
import sys
import argparse
import matplotlib.pyplot as plt
import skimage.transform

import theano
import theano.tensor as T
import lasagne

from lasagne.utils import floatX
from lasagne.layers import InputLayer, ConcatLayer
from lasagne.layers import Conv2DLayer as ConvLayer

def prep_image(im):
    MEAN_VALUES = np.array([104, 117, 123]).reshape((3,1,1))

    if len(im.shape) == 2:
        im = im[:, :, np.newaxis]
        im = np.repeat(im, 3, axis=2)
    h, w, _ = im.shape

    if h < w:
        im = skimage.transform.resize(im, (IMAGE_W, int(w*IMAGE_W/h)), preserve_range=True)
    else:
        im = skimage.transform.resize(im, (int(h*IMAGE_W/w), IMAGE_W), preserve_range=True)

    # Central crop
    h, w, _ = im.shape
    im = im[h//2-IMAGE_W//2:h//2+IMAGE_W//2, w//2-IMAGE_W//2:w//2+IMAGE_W//2]

    rawim = np.copy(im).astype('uint8')

    # Shuffle axes to c01
    im = np.swapaxes(np.swapaxes(im, 1, 2), 0, 1)

    # Convert RGB to BGR
    im = im[::-1, :, :]

    im = im - MEAN_VALUES
    return rawim, floatX(im[np.newaxis])

def deprocess(x):
    MEAN_VALUES = np.array([104, 117, 123]).reshape((3,1,1))
    x = np.copy(x[0])
    x += MEAN_VALUES

    x = x[::-1]
    x = np.swapaxes(np.swapaxes(x, 0, 1), 1, 2)

    x = np.clip(x, 0, 255).astype('uint8')
    return x

def gram_matrix(x):
    x = x.flatten(ndim=3)
    g = T.tensordot(x, x, axes=([2], [2]))
    return g

def style_loss(A, X, layer):
    a = A[layer]
    x = X[layer]

    A = gram_matrix(a)
    G = gram_matrix(x)

    N = a.shape[1]
    M = a.shape[2] * a.shape[3]

    loss = 1./(4 * N**2 * M**2) * ((G - A)**2).sum()
    return loss

def style_loss_relative(A, X, layer):
    a = A[layer]
    x = X[layer]

    A = gram_matrix(a)
    G = gram_matrix(x)

    loss = ((G - A)**2).sum() / (G**2).sum()
    return loss

def build_model_one_scale(n_feature_maps, filter_size):
    net = {}
    net['input'] = InputLayer((1, 3, IMAGE_W, IMAGE_W))
    net['conv1_1'] = ConvLayer(net['input'], n_feature_maps, filter_size, pad=filter_size//2, flip_filters=False)
    return net

def build_model_multiscale(n_feature_maps, scales, nonlinearity=lasagne.nonlinearities.rectify):
    net = {}
    net['input'] = InputLayer((1, 3, IMAGE_W, IMAGE_W))

    multiple_scales = [ConvLayer(net['input'], n_feature_maps, filter_size, pad=filter_size//2, flip_filters=False,
                                 nonlinearity=nonlinearity)
                       for filter_size in scales]
    net['conv1_1'] = ConcatLayer(multiple_scales)
    return net

def optimize(texture_file, net, n_iter, scales):
    texture = plt.imread(texture_file)
    rawim, texture = prep_image(texture)

    layers = ['conv1_1']
    layers = {k: net[k] for k in layers}

    input_im_theano = T.tensor4()
    outputs = lasagne.layers.get_output(layers.values(), input_im_theano)
    texture_features = {k: theano.shared(output.eval({input_im_theano: texture}))
                        for k, output in zip(layers.keys(), outputs)}

    generated_image = theano.shared(floatX(np.random.uniform(-1, 1, (1, 3, IMAGE_W, IMAGE_W))))
    gen_features = lasagne.layers.get_output(layers.values(), generated_image)
    gen_features = {k: v for k, v in zip(layers.keys(), gen_features)}

    losses = []
    losses_test = []
    losses.append(1e7 * style_loss(texture_features, gen_features, 'conv1_1'))
    losses_test.append(1 * style_loss_relative(texture_features, gen_features, 'conv1_1'))
    total_loss = sum(losses)
    total_loss_test = sum(losses_test)

    grad = T.grad(total_loss, generated_image)
    f_loss = theano.function([], total_loss)
    f_test_loss = theano.function([], total_loss_test)
    f_grad = theano.function([], grad)

    def eval_loss(x0):
        x0 = floatX(x0.reshape((1, 3, IMAGE_W, IMAGE_W)))
        generated_image.set_value(x0)
        return f_loss().astype('float64')

    def test_loss(x0):
        x0 = floatX(x0.reshape((1, 3, IMAGE_W, IMAGE_W)))
        generated_image.set_value(x0)
        return f_test_loss().astype('float64')

    def eval_grad(x0):
        x0 = floatX(x0.reshape((1, 3, IMAGE_W, IMAGE_W)))
        generated_image.set_value(x0)
        return np.array(f_grad()).flatten().astype('float64')

    texture_init = np.random.uniform(-1, 1, (1, 3, IMAGE_W, IMAGE_W))

    generated_image.set_value(floatX(texture_init))
    x0 = generated_image.get_value().astype('float64')
    xs = []
    xs.append(x0)

    scipy.optimize.fmin_l_bfgs_b(eval_loss, x0.flatten(), fprime=eval_grad, maxfun=n_iter)
    x0 = generated_image.get_value().astype('float64')

    return x0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--texture', required=True, type=str, help='Path to the reference texture')
    parser.add_argument('-s', '--size', required=True, type=int, help='Size of a synthesised texture in pixels')
    parser.add_argument('-f', '--target-file', default=None, type=str, help='File name of a syntesised texture')
    parser.add_argument('-n', '--n-iter', default=4000, type=int, help='Number of L-BFGS optinisation iterations')
    parser.add_argument('-c', '--n-features', default=128, type=int, help='Number of feature maps per each scale')
    parser.add_argument('--scales', default=None, type=int, nargs='*', help='Sizes of convolutional filters')
    parser.add_argument('-l', '--linear', action='store_true', help='Use linear model (conv layer without non-linearity)')
    args = parser.parse_args()

    texture_file = args.texture

    global IMAGE_W
    IMAGE_W = args.size

    n_iter = args.n_iter
    n_feature_maps = args.n_features

    # if no scales provided, assume multiscale model
    if args.scales is None:
        scales = [3, 5, 7, 11, 15, 23, 37, 55]
    else:
        scales = args.scales

    if not args.linear:
        net = build_model_multiscale(n_feature_maps, scales)
    else:
        net = build_model_multiscale(n_feature_maps, scales, nonlinearity=None)

    synthesised = optimize(texture_file, net, n_iter, scales)

    if args.target_file is None:
        target_file = texture_file.split('.')[0] + '_synthesised.jpg'
    else:
        target_file = args.target_file
    plt.imsave(target_file, deprocess(synthesised))

    return 0

if __name__ == '__main__':
    main()
