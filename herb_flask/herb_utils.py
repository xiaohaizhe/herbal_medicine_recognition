# coding: utf-8

# plant_utils.py

import numpy as np
# import cv2
import os
import logging
import PIL.Image as Image


def check_img_dir(file_dir):
    isExists = os.path.exists(file_dir)
    if not isExists:
        print('图片上传目录不存在')
        os.makedirs(file_dir)
        print('图片上传目录已创建')


def img_resize(imgpath, img_size):
    # format image
        img = Image.open(imgpath)
        img = img.convert("RGB")
        '''
        # use opencv
        if (img.width > img.height):
            scale = float(img_size) / float(img.height)
            img = np.array(cv2.resize(np.array(img), (
            int(img.width * scale + 1), img_size))).astype(np.float32)
        else:
            scale = float(img_size) / float(img.width)
            img = np.array(cv2.resize(np.array(img), (
            img_size, int(img.height * scale + 1)))).astype(np.float32)
        '''
        # use PIL
        if (img.width > img.height):
            scale = float(img_size) / float(img.height)
            img = np.array(img.resize((int(img.width * scale + 1), img_size),
                    Image.ANTIALIAS)).astype(np.float32)
        else:
            scale = float(img_size) / float(img.width)
            img = np.array(img.resize((img_size, int(img.height * scale + 1)),
                    Image.ANTIALIAS)).astype(np.float32)
        img = (img[
                  (img.shape[0] - img_size) // 2:
                  (img.shape[0] - img_size) // 2 + img_size,
                  (img.shape[1] - img_size) // 2:
                  (img.shape[1] - img_size) // 2 + img_size,
                  :]-127)/255
        return img


def train_log(filename='logfile'):
    # create logger
    logger_name = "filename"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)

    # create file handler
    log_path = './' + filename + '.log'
    fh = logging.FileHandler(log_path)
    ch = logging.StreamHandler()

    # create formatter
    fmt = "%(asctime)-15s %(levelname)s %(filename)s %(lineno)d %(process)d %(message)s"
    datefmt = "%a %d %b %Y %H:%M:%S"
    formatter = logging.Formatter(fmt, datefmt)

    # add handler and formatter to logger
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger
