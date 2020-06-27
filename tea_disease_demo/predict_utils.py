# coding: utf-8

# plant_utils.py

import numpy as np
# import cv2
import os
import logging
import random
import PIL.Image as Image
import PIL.ExifTags as ExifTags


def check_img_dir(file_dir):
    isExists = os.path.exists(file_dir)
    if not isExists:
        print('图片上传目录不存在')
        os.makedirs(file_dir)
        print('图片上传目录已创建')


def img_resize(imgpath, img_size):
    # format image
        img = Image.open(imgpath)
        if(img_size == -1):
            # img_size为-1不裁剪
            return img
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


def get_exif_value(image, exiftype):
    '''
    获取图片exif信息中的对应字段
        image : PIL.Image 图片文件
        exiftype : exif标签名称
    返回值：ret --> int
        -1：读取失败

    '''
    ret = -1
    if (hasattr(image, '_getexif') and image._getexif()):
        exifinfo = {ExifTags.TAGS[k]: v for k, v in image._getexif().items() if k in ExifTags.TAGS}
        # print(exifinfo)
        ret = exifinfo.get(exiftype, -1)

    return ret


def get_pic_index(first):
    ret = []
    indexList = list(range(0,11))
    if (first in indexList):
        a = first
    else:
        a = random.randint(0, 10)
    ret.append(a)
    for i in range(3):
        indexList.remove(a)
        a = random.choice(indexList)
        ret.append(a)
    return ret


def get_pic_prob():
    ret = []
    ret.append(random.randint(5999, 9999))
    for i in range(3):
        left = 10000 - sum(ret)
        temp = random.randint(0, left)
        ret.append(temp)
    ret.sort(reverse=True)

    return ret


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
