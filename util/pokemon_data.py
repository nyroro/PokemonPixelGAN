# coding: utf-8
import os
from .img_slice import crop_all_img
import imageio
import numpy as np
import matplotlib.pyplot as plt
IMG_PATH = 'pokemon'
OUTPUT_PATH = 'output'


def read_data():
    if not os.path.exists(IMG_PATH):
        crop_all_img(IMG_PATH)

    ret = []

    for file in os.listdir(IMG_PATH):
        filepath = os.path.join(IMG_PATH, file)
        data = imageio.imread(filepath)
        ret.append(data)
    ret = np.array(ret)
    ret = ret / 255
    ret[:, :, :, 0] += 1.0 - ret[:, :, :, 3]

    ret[:, :, :, 1] += 1.0 - ret[:, :, :, 3]

    ret[:, :, :, 2] += 1.0 - ret[:, :, :, 3]

    ret = ret[:, :, :, 0:3]
    ret = np.clip(ret, 0., 1.)
    print('Images data shape', ret.shape)
    return ret


def sample_image(datas, r, c, name):
    assert r * c == len(datas), "datas size not equal to r*c"
    gen_imgs = datas
    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i, j].imshow(gen_imgs[cnt])
            axs[i, j].axis('off')
            cnt += 1
    if not os.path.exists(OUTPUT_PATH):
        os.mkdir(OUTPUT_PATH)
    fig.savefig("%s/%s.png" % (OUTPUT_PATH, name))
    plt.close()
