from keras.models import Sequential, Model
from keras.layers import *
from keras.layers.advanced_activations import LeakyReLU
from keras.activations import relu
from keras.initializers import RandomNormal

from instance_normalization import InstanceNormalization

from keras.applications import *
import tensorflow as tf
import matplotlib.pyplot as plt
import keras.backend as K
from keras.optimizers import RMSprop, SGD, Adam
from PIL import Image
import numpy as np
from random import randint, shuffle
import h5py as h5
import scipy.misc
import os

channel_axis = -1
channel_first = False
use_instancenorm = True
use_lsgan = False
use_nsgan = False  # non-saturating GAN
isRGB = True



def __conv_init(a):
    print("conv_init", a)
    k = RandomNormal(0, 0.02)(a)  # for convolution kernel
    k.conv_weight = True
    return k


conv_init = RandomNormal(0, 0.02)
gamma_init = RandomNormal(1., 0.02)  # for batch normalization


# Basic discriminator
def conv2d(f, *a, **k):
    return Conv2D(f, kernel_initializer=conv_init, *a, **k)


def batchnorm():
    return BatchNormalization(momentum=0.9, axis=channel_axis, epsilon=1.01e-5,
                              gamma_initializer=gamma_init)


def instance_norm():
    return InstanceNormalization(axis=channel_axis, epsilon=1.01e-5,
                                 gamma_initializer=gamma_init)


def BASIC_D(nc_in, ndf, max_layers=3, use_sigmoid=True):
    """DCGAN_D(nc, ndf, max_layers=3)
       nc: channels
       ndf: filters of the first layer
       max_layers: max hidden layers
    """
    if channel_first:
        input_a = Input(shape=(nc_in, None, None))
    else:
        input_a = Input(shape=(None, None, nc_in))
    _ = input_a
    _ = conv2d(ndf, kernel_size=4, strides=2, padding="same", name='First')(_)
    _ = LeakyReLU(alpha=0.2)(_)

    for layer in range(1, max_layers):
        out_feat = ndf * min(2 ** layer, 8)
        _ = conv2d(out_feat, kernel_size=4, strides=2, padding="same",
                   use_bias=False, name='pyramid.{0}'.format(layer)
                   )(_)
        _ = batchnorm()(_, training=1)
        _ = LeakyReLU(alpha=0.2)(_)

    out_feat = ndf * min(2 ** max_layers, 8)
    _ = ZeroPadding2D(1)(_)
    _ = conv2d(out_feat, kernel_size=4, use_bias=False, name='pyramid_last')(_)
    _ = batchnorm()(_, training=1)
    _ = LeakyReLU(alpha=0.2)(_)

    # final layer
    _ = ZeroPadding2D(1)(_)
    _ = conv2d(1, kernel_size=4, name='final'.format(out_feat, 1),
               activation="sigmoid" if use_sigmoid else None)(_)
    return Model(inputs=[input_a], outputs=_)


def UNET_G(isize, nc_in=3, nc_out=3, ngf=64, fixed_input_size=True, use_batchnorm=True):
    s = isize if fixed_input_size else None
    _ = inputs = Input(shape=(s, s, nc_in))
    x_i = Lambda(lambda x: x[:, :, :, 0:3], name='x_i')(inputs)
    y_i = Lambda(lambda x: x[:, :, :, 4:7], name='y_j')(inputs)
    xi_and_y_i = concatenate([x_i, y_i], name='xi_yi')
    xi_yi_sz64 = AveragePooling2D(pool_size=2)(xi_and_y_i)
    xi_yi_sz32 = AveragePooling2D(pool_size=4)(xi_and_y_i)
    xi_yi_sz16 = AveragePooling2D(pool_size=8)(xi_and_y_i)
    xi_yi_sz8 = AveragePooling2D(pool_size=16)(xi_and_y_i)
    layer1 = conv2d(64, kernel_size=4, strides=2, use_bias=(not (use_batchnorm and s > 2)),
                    padding="same", name='layer1')(_)
    layer1 = LeakyReLU(alpha=0.2)(layer1)
    layer1 = concatenate([layer1, xi_yi_sz64])  # ==========
    layer2 = conv2d(128, kernel_size=4, strides=2, use_bias=(not (use_batchnorm and s > 2)),
                    padding="same", name='layer2')(layer1)
    if use_instancenorm:
        layer2 = instance_norm()(layer2, training=1)
    else:
        layer2 = batchnorm()(layer2, training=1)
    layer3 = LeakyReLU(alpha=0.2)(layer2)
    layer3 = concatenate([layer3, xi_yi_sz32])  # ==========
    layer3 = conv2d(256, kernel_size=4, strides=2, use_bias=(not (use_batchnorm and s > 2)),
                    padding="same", name='layer3')(layer3)
    if use_instancenorm:
        layer3 = instance_norm()(layer3, training=1)
    else:
        layer3 = batchnorm()(layer3, training=1)
    layer4 = LeakyReLU(alpha=0.2)(layer3)
    layer4 = concatenate([layer4, xi_yi_sz16])  # ==========
    layer4 = conv2d(512, kernel_size=4, strides=2, use_bias=(not (use_batchnorm and s > 2)),
                    padding="same", name='layer4')(layer4)
    if use_instancenorm:
        layer4 = instance_norm()(layer4, training=1)
    else:
        layer4 = batchnorm()(layer4, training=1)
    layer4 = LeakyReLU(alpha=0.2)(layer4)
    layer4 = concatenate([layer4, xi_yi_sz8])  # ==========

    layer9 = Conv2DTranspose(256, kernel_size=4, strides=2, use_bias=not use_batchnorm,
                             kernel_initializer=conv_init, name='layer9')(layer4)
    layer9 = Cropping2D(((1, 1), (1, 1)))(layer9)
    if use_instancenorm:
        layer9 = instance_norm()(layer9, training=1)
    else:
        layer9 = batchnorm()(layer9, training=1)
    layer9 = Concatenate(axis=channel_axis)([layer9, layer3])
    layer9 = Activation('relu')(layer9)
    layer9 = concatenate([layer9, xi_yi_sz16])  # ==========
    layer10 = Conv2DTranspose(128, kernel_size=4, strides=2, use_bias=not use_batchnorm,
                              kernel_initializer=conv_init, name='layer10')(layer9)
    layer10 = Cropping2D(((1, 1), (1, 1)))(layer10)
    if use_instancenorm:
        layer10 = instance_norm()(layer10, training=1)
    else:
        layer10 = batchnorm()(layer10, training=1)
    layer10 = Concatenate(axis=channel_axis)([layer10, layer2])
    layer10 = Activation('relu')(layer10)
    layer10 = concatenate([layer10, xi_yi_sz32])  # ==========
    layer11 = Conv2DTranspose(64, kernel_size=4, strides=2, use_bias=not use_batchnorm,
                              kernel_initializer=conv_init, name='layer11')(layer10)
    layer11 = Cropping2D(((1, 1), (1, 1)))(layer11)
    if use_instancenorm:
        layer11 = instance_norm()(layer11, training=1)
    else:
        layer11 = batchnorm()(layer11, training=1)
    layer11 = Activation('relu')(layer11)

    layer12 = concatenate([layer11, xi_yi_sz64])  # ==========
    layer12 = Activation('relu')(layer12)
    layer12 = Conv2DTranspose(32, kernel_size=4, strides=2, use_bias=not use_batchnorm,
                              kernel_initializer=conv_init, name='layer12')(layer12)
    layer12 = Cropping2D(((1, 1), (1, 1)))(layer12)
    if use_instancenorm:
        layer12 = instance_norm()(layer12, training=1)
    else:
        layer12 = batchnorm()(layer12, training=1)

    layer12 = conv2d(4, kernel_size=4, strides=1, use_bias=(not (use_batchnorm and s > 2)),
                     padding="same", name='out128')(layer12)

    m_g = Lambda(lambda x: x[:, :, :, 0:1], name='mask')(layer12)
    x_i_j = Lambda(lambda x: x[:, :, :, 1:], name='x_i_j')(layer12)
    m_g = Activation("sigmoid", name='mask_sigmoid')(m_g)
    x_i_j = Activation("tanh", name='x_i_j_tanh')(x_i_j)
    out = concatenate([x_i_j, m_g], name='out128_concat')

    return Model(inputs=inputs, outputs=[out])


def cycle_variables(netG1):
    """
    Intermidiate params:
        x_i: human w/ cloth i, shape=(128,96,3)
        y_i: stand alone cloth i, shape=(128,96,3)
        y_j: stand alone cloth j, shape=(128,96,3)
        alpha: mask for x_i_j, shape=(128,96,1)
        x_i_j: generated fake human swapping cloth i to j, shape=(128,96,3)

    Out:
        real_input: concat[x_i, y_i, y_j], shape=(128,96,9)
        fake_output: masked_x_i_j = alpha*x_i_j + (1-alpha)*x_i, shape=(128,96,3)
        rec_input: output of the second generator (generating image similar to x_i), shape=(128,96,3)
        fn_generate: a path from input to G_out and cyclic G_out
    """
    real_input = netG1.inputs[0]
    fake_out = netG1.outputs[0]
    # Legacy: how to split channels
    # https://github.com/fchollet/keras/issues/5474
    x_i = Lambda(lambda x: x[:, :, :, 0:4])(real_input)
    im_i = Lambda(lambda x: x[:, :, :, 0:3])(real_input)

    m_g = Lambda(lambda x: x[:, :, :, 3:])(fake_out)
    im_i_j = Lambda(lambda x: x[:, :, :, 0:3])(fake_out)
    fake_im = m_g * im_i_j + (1 - m_g) * im_i
    fake_output = concatenate([fake_im, m_g], axis = -1)

    concat_input_G2 = concatenate([fake_output, x_i], axis=-1)  # swap

    rec_input = netG1([concat_input_G2])
    rec_m_g = Lambda(lambda x: x[:, :, :, 3:])(rec_input)
    rec_i_j= Lambda(lambda x: x[:, :, :, 0:3])(rec_input)
    rec_im = rec_m_g * rec_i_j + (1 - rec_m_g) * fake_im

    rec_input = concatenate([rec_im, rec_m_g], axis = -1)

    fn_generate = K.function([real_input], [fake_output, rec_input])
    return real_input, fake_out, rec_input, fn_generate, m_g


def loss_fn(output, target):
    if use_lsgan:
        return K.mean(K.abs(K.square(output-target)))
    else:
        return -K.mean(K.log(output+1e-12)*target+K.log(1-output+1e-12)*(1-target))


def loss_fn_mask(output, target):
    return K.mean(K.abs(output - target))


def D_loss(netD, real, fake, rec):
    # x_i, y_i, y_j = tf.split(real, [3, 3, 3], 3)
    x_i = Lambda(lambda x: x[:, :, :, 0:3])(real)
    m_i = Lambda(lambda x: x[:, :, :, 3:4])(real)
    x_j = Lambda(lambda x: x[:, :, :, 4:7])(real)
    m_j = Lambda(lambda x: x[:, :, :, 7:8])(real)

    im = Lambda(lambda x: x[:, :, :, 0:3])(fake)
    m_g = Lambda(lambda x: x[:, :, :, 3:])(fake)

    rec_im = Lambda(lambda x: x[:, :, :, 0:3])(rec)

    output_real = netD(concatenate([x_i, x_i*m_i], axis = -1))  # positive sample
    output_fake = netD(concatenate([im, x_j*m_j], axis = -1))  # negative sample
    output_fake2 = netD(concatenate([x_i, x_j*m_j], axis = -1))  # negative sample 2

    loss_D_real = loss_fn(output_real, K.ones_like(output_real))
    loss_D_fake = loss_fn(output_fake, K.zeros_like(output_fake))
    loss_D_fake2 = loss_fn(output_fake2, K.zeros_like(output_fake2))  # New loss term for discriminator

    loss_masks = loss_fn_mask(m_g, m_i)

    loss_G = loss_fn(output_fake, K.ones_like(output_fake)) + loss_masks

    loss_D = loss_D_real + (loss_D_fake + loss_D_fake2)
    loss_cyc = K.mean(K.abs(rec_im - x_i))  # cycle loss
    return loss_D, loss_G, loss_cyc


def load_data(dataset):
    with h5.File(dataset, 'r') as f:
        train_images = f['ih'][:]
        train_masks = f['b_'][:]
        train_images = train_images.transpose(0, 3, 2, 1)
        train_masks = train_masks.transpose(0, 3, 2, 1)

        # choose images with upper body
        length = len(train_images)
        idx_upper_body_only = []
        for i in range(length):
            if 3 in train_masks[i,:,:,:]:
                idx_upper_body_only.append(i)
        train_images = train_images[idx_upper_body_only]
        train_masks = train_masks[idx_upper_body_only]

        # make upper body white, and elsewhere black
        train_masks[train_masks != 3] = 0
        train_masks[train_masks == 3] = 1

        # concatenate images and masks
        train = np.concatenate([train_images, train_masks], -1)

        return train


def read_image(data, idx_i):

    length = len(data)
    # Load consumer picture
    im = data[idx_i,:,:,:]
    img_x_i = im

    # Load model picture randomly
    idx_j = np.random.choice(length)
    while idx_j == idx_i:
        idx_j = np.random.choice(length)
    im = data[idx_j,:,:,:]
    img_x_j = im

    img = np.concatenate([img_x_i, img_x_j], axis=-1)
    assert img.shape[-1] == 8

    return img


def minibatch(data, batchsize):
    length = len(data)
    epoch = i = 0
    tmpsize = None
    np.random.shuffle(data)
    while True:
        size = tmpsize if tmpsize else batchsize
        if i+size > length:
            np.random.shuffle(data)
            i = 0
            epoch+=1
        rtn = [read_image(data, j) for j in range(i,i+size)]
        i+=size
        tmpsize = yield epoch, np.float32(rtn)


def minibatchAB(dataA, batchsize):
    batchA=minibatch(dataA, batchsize)
    tmpsize = None
    while True:
        ep1, A = batchA.send(tmpsize)
        tmpsize = yield ep1, A


def showX(X):
    length = len(X)
    if not os.path.exists('samples'):
        os.makedirs('samples')
    for i in range(length):
        consumer_im = X[i,:,:,:3]
        save_images('samples/consumer_%s.png' % i, consumer_im)
        model_im = X[i,:,:,4:7]

        save_images('samples/model_%s.png' % i, model_im)



def showG(cycleA_generate, A):
    if not os.path.exists('samples'):
        os.makedirs('samples')
    # def G(fn_generate, X):
    #     r = np.array([fn_generate([X[i:i+1]]) for i in range(X.shape[0])])
    #     return r.swapaxes(0,1)[:,:,0]
    # rA = G(cycleA_generate, A)
    # fake_output = rA[0]
    # length = len(fake_output)
    # for i in range(length):
    #     fake_im = fake_output[i,:,:,2:]
    #     save_images('samples/fake_%s.png' % i, fake_im)
    length = len(A)
    rA = cycleA_generate([A])
    fake_output = rA[0]
    print(fake_output.shape)
    for i in range(length):
        fake_im = fake_output[i,:,:,1:]
        save_images('samples/fake_%s.png' % i, fake_im)


def save_images(path, image):
    return scipy.misc.imsave(path, image)

