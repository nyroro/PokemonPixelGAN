# coding: utf-8
from PIL import Image
import os
POKEMON_ALL_IMAGE_PATH = 'pokemon_all.png'
SAVE_FOLDER = 'pokemon'
DELETE_FILE = ['1240_1240', '1200_1240']


def crop_all_img(save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    crop_size = (40, 40)
    img = Image.open(POKEMON_ALL_IMAGE_PATH)
    size = img.size
    for x in range(0, size[0], crop_size[0]):
        for y in range(0, size[1], crop_size[1]):
            region = img.crop((x, y, x + crop_size[0], y + crop_size[1]))
            region.save('%s/%d_%d.png' % (save_path, x, y))
    for del_file in DELETE_FILE:
        file = '%s/%s.png' % (save_path, del_file)
        if os.path.exists(file):
            os.remove(file)


    