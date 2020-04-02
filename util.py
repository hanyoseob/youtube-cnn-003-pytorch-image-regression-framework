import os
import numpy as np
from scipy.stats import poisson
from skimage.transform import rescale, resize

import torch
import torch.nn as nn

## 네트워크 저장하기
def save(ckpt_dir, net, optim, epoch):
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    torch.save({'net': net.state_dict(), 'optim': optim.state_dict()},
               "%s/model_epoch%d.pth" % (ckpt_dir, epoch))

## 네트워크 불러오기
def load(ckpt_dir, net, optim):
    if not os.path.exists(ckpt_dir):
        epoch = 0
        return net, optim, epoch

    ckpt_lst = os.listdir(ckpt_dir)
    ckpt_lst.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

    dict_model = torch.load('%s/%s' % (ckpt_dir, ckpt_lst[-1]))

    net.load_state_dict(dict_model['net'])
    optim.load_state_dict(dict_model['optim'])
    epoch = int(ckpt_lst[-1].split('epoch')[1].split('.pth')[0])

    return net, optim, epoch

## Add Sampling
def add_sampling(img, type="random", opts=None):
    sz = img.shape

    if type == "uniform":
        ds_y = opts[0].astype(np.int)
        ds_x = opts[1].astype(np.int)

        msk = np.zeros(img.shape)
        msk[::ds_y, ::ds_x, :] = 1

        dst = img * msk

    elif type == "random":
        prob = opts[0]

        # rnd = np.random.rand(sz[0], sz[1], 1)
        # msk = (rnd < prob).astype(np.float)
        # msk = np.tile(msk, (1, 1, sz[2]))

        rnd = np.random.rand(sz[0], sz[1], sz[2])
        msk = (rnd > prob).astype(np.float)

        dst = img * msk

    elif type == "gaussian":
        x0 = opts[0]
        y0 = opts[1]
        sgmx = opts[2]
        sgmy = opts[3]

        a = opts[4]

        ly = np.linspace(-1, 1, sz[0])
        lx = np.linspace(-1, 1, sz[1])

        x, y = np.meshgrid(lx, ly)

        gaus = a * np.exp(-((x - x0)**2/(2*sgmx**2) + (y - y0)**2/(2*sgmy**2)))
        gaus = np.tile(gaus[:, :, np.newaxis], (1, 1, sz[2]))
        rnd = np.random.rand(sz[0], sz[1], sz[2])
        msk = (rnd < gaus).astype(np.float)

        # gaus = a * np.exp(-((x - x0) ** 2 / (2 * sgmx ** 2) + (y - y0) ** 2 / (2 * sgmy ** 2)))
        # gaus = np.tile(gaus[:, :, np.newaxis], (1, 1, 1))
        # rnd = np.random.rand(sz[0], sz[1], 1)
        # msk = (rnd < gaus).astype(np.float)
        # msk = np.tile(msk, (1, 1, sz[2]))

        dst = img * msk

    return dst

## Add Noise
def add_noise(img, type="random", opts=None):
    sz = img.shape

    if type == "random":
        sgm = opts[0]

        noise = sgm / 255.0 * np.random.randn(sz[0], sz[1], sz[2])

        dst = img + noise

    elif type == "poisson":
        dst = poisson.rvs(255.0 * img) / 255.0
        noise = dst - img

    return dst

## Add blurring
def add_blur(img, type="bilinear", opts=None):
    if type == "nearest":
        order = 0
    elif type == "bilinear":
        order = 1
    elif type == "biquadratic":
        order = 2
    elif type == "bicubic":
        order = 3
    elif type == "biquartic":
        order = 4
    elif type == "biquintic":
        order = 5

    sz = img.shape
    if len(opts) == 1:
        keepdim = True
    else:
        keepdim = opts[1]

    # dw = 1.0 / opts[0]
    # dst = rescale(img, scale=(dw, dw, 1), order=order)
    dst = resize(img, output_shape=(sz[0] // opts[0], sz[1] // opts[0], sz[2]), order=order)

    if keepdim:
        # dst = rescale(dst, scale=(1 / dw, 1 / dw, 1), order=order)
        dst = resize(dst, output_shape=(sz[0], sz[1], sz[2]), order=order)

    return dst