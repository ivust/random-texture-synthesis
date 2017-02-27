import numpy as np
import pickle
import scipy
import sys
import os
import matplotlib.pyplot as plt
import skimage.transform

sys.path.append('/master/')

import theano
import theano.tensor as T
import lasagne

from lasagne.utils import floatX
from lasagne.layers import InputLayer, ConcatLayer
from lasagne.layers import Conv2DLayer as ConvLayer
from lasagne.layers import Pool2DLayer as PoolLayer

def prep_image(im):
    MEAN_VALUES = np.array([104, 117, 123]).reshape((3,1,1))

    if len(im.shape) == 2:
        im = im[:, :, np.newaxis]
        im = np.repeat(im, 3, axis=2)
    if im.shape[2] == 4:
        im = im[:,:,:3]
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

def deprocess(x):
    MEAN_VALUES = np.array([104, 117, 123]).reshape((3,1,1))
    x = np.copy(x[0])
    x += MEAN_VALUES

    x = x[::-1]
    x = np.swapaxes(np.swapaxes(x, 0, 1), 1, 2)
                    
    x = np.clip(x, 0, 255).astype('uint8')
    return x

def style_loss_relative(A, X, layer, layers):
    a = A[layer]
    x = X[layer]
                                                                                
    G_all = []
    for l in layers:
        G_layer = gram_matrix(A[l])
        G_all.append((G_layer**2).mean())
    G_all = T.sum(G_all)

    A = gram_matrix(a)
    G = gram_matrix(x)

    # loss = ((G - A)**2).mean() / (A**2).mean()
    loss = ((G - A)**2).mean() / G_all
    return loss

def build_model():
    net = {}
    net['input'] = InputLayer((1, 3, IMAGE_W, IMAGE_W))
    net['conv1_1'] = ConvLayer(net['input'], 64, 3, pad=1, flip_filters=False)
    net['conv1_2'] = ConvLayer(net['conv1_1'], 64, 3, pad=1, flip_filters=False)
    net['pool1'] = PoolLayer(net['conv1_2'], 2, mode='average_exc_pad')
    net['conv2_1'] = ConvLayer(net['pool1'], 128, 3, pad=1, flip_filters=False)
    net['conv2_2'] = ConvLayer(net['conv2_1'], 128, 3, pad=1, flip_filters=False)
    net['pool2'] = PoolLayer(net['conv2_2'], 2, mode='average_exc_pad')
    net['conv3_1'] = ConvLayer(net['pool2'], 256, 3, pad=1, flip_filters=False)
    net['conv3_2'] = ConvLayer(net['conv3_1'], 256, 3, pad=1, flip_filters=False)
    net['conv3_3'] = ConvLayer(net['conv3_2'], 256, 3, pad=1, flip_filters=False)
    net['conv3_4'] = ConvLayer(net['conv3_3'], 256, 3, pad=1, flip_filters=False)
    net['pool3'] = PoolLayer(net['conv3_4'], 2, mode='average_exc_pad')
    net['conv4_1'] = ConvLayer(net['pool3'], 512, 3, pad=1, flip_filters=False)
    net['conv4_2'] = ConvLayer(net['conv4_1'], 512, 3, pad=1, flip_filters=False)
    net['conv4_3'] = ConvLayer(net['conv4_2'], 512, 3, pad=1, flip_filters=False)
    net['conv4_4'] = ConvLayer(net['conv4_3'], 512, 3, pad=1, flip_filters=False)
    net['pool4'] = PoolLayer(net['conv4_4'], 2, mode='average_exc_pad')
    net['conv5_1'] = ConvLayer(net['pool4'], 512, 3, pad=1, flip_filters=False)
    net['conv5_2'] = ConvLayer(net['conv5_1'], 512, 3, pad=1, flip_filters=False)
    net['conv5_3'] = ConvLayer(net['conv5_2'], 512, 3, pad=1, flip_filters=False)
    net['conv5_4'] = ConvLayer(net['conv5_3'], 512, 3, pad=1, flip_filters=False)
    net['pool5'] = PoolLayer(net['conv5_4'], 2, mode='average_exc_pad')
    return net

def evaluate(texture_file, texture_file_synthesized, net):
    texture = plt.imread(texture_file) 
    rawim, texture = prep_image(texture)

    texture_synt = plt.imread(texture_file_synthesized) 
    rawim_synt, texture_synt = prep_image(texture_synt)

    layers = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']
    layers = {k: net[k] for k in layers}

    input_im_theano = T.tensor4()
    outputs = lasagne.layers.get_output(layers.values(), input_im_theano)
    texture_features = {k: theano.shared(output.eval({input_im_theano: texture}))
                        for k, output in zip(layers.keys(), outputs)}

    generated_image = theano.shared(floatX(texture_synt))
    gen_features = lasagne.layers.get_output(layers.values(), generated_image)
    gen_features = {k: v for k, v in zip(layers.keys(), gen_features)}

    losses = []
    losses_test = []
    layers_all = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']
    losses.append(1 * style_loss(texture_features, gen_features, 'conv1_1'))
    losses.append(1 * style_loss(texture_features, gen_features, 'conv2_1'))
    losses.append(1 * style_loss(texture_features, gen_features, 'conv3_1'))
    losses.append(1 * style_loss(texture_features, gen_features, 'conv4_1'))
    losses.append(1 * style_loss(texture_features, gen_features, 'conv5_1'))
    losses_test.append(0.2 * style_loss_relative(texture_features, gen_features, 'conv1_1', layers_all))
    losses_test.append(0.2 * style_loss_relative(texture_features, gen_features, 'conv2_1', layers_all))
    losses_test.append(0.2 * style_loss_relative(texture_features, gen_features, 'conv3_1', layers_all))
    losses_test.append(0.2 * style_loss_relative(texture_features, gen_features, 'conv4_1', layers_all))
    losses_test.append(0.2 * style_loss_relative(texture_features, gen_features, 'conv5_1', layers_all))
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

    generated_image.set_value(floatX(texture_synt)) 
    x0 = generated_image.get_value().astype('float64')
    xs = []
    xs.append(x0)
    losses = []
    test_losses = []
    losses.append(eval_loss(x0))
    test_losses.append(test_loss(x0))

    return (losses[-1], test_losses[-1])

def main(argv):
    texture_file = argv[0]
    texture_file_synthesized = argv[1]

    global IMAGE_W
    IMAGE_W = int(argv[2])

    net = build_model()
    values = pickle.load(open('/master/parameters/vgg19_normalized.pkl', 'rb'), encoding='latin1')['param values']
    lasagne.layers.set_all_param_values(net['pool5'], values)

    loss, relative_loss = evaluate(texture_file, texture_file_synthesized, net)
    return (float(loss), float(relative_loss))

if __name__ == '__main__':
    main(sys.argv[1:])
