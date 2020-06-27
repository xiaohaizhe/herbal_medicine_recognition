# coding: utf-8
# 将分目录图像文件转为标签文件
# 标签ID为目录名称，图片ID为文件名
# 注意：当前支持目录名称为整数
# 同时生成标签json文件，文件内容形如：
# [{'label_id': 0, 'image_id': 'IMG_0785_DOWN_TB.JPG'}, {'label_id': 0, 'image_id': 'IMG_0785_DOWN_LR.JPG'}]
# format images to label file

import os
import random
import json
import shutil
import math
import pics_augmetation

TRAIN_RATIO = 0.8
TRAIN_DIR = './train_pics/'
VAL_DIR = './val_pics/'
ENTIRE_LABELFILE = './all_labels.json'


def files2label(src_dir):
    '''

    Args:
        src_dir: 源目录

    Returns:

    '''
    if not src_dir: return None

    total_labels = []
    dir_names = os.listdir(src_dir)
    for pic_dir in dir_names:
        # print(pic_dir, type(pic_dir))
        if os.path.isdir(src_dir + str(pic_dir)):
            for name in os.listdir(src_dir + str(pic_dir)):
                temp_dict = {}
                temp_dict['label_id'] = int(pic_dir)
                temp_dict['image_id'] = name
                total_labels.append(temp_dict)

    return total_labels


def save_labels(labels, label_file):
    '''
    将文件lebel信息固化到文件中
    Args:
        labels: 标签信息
        dstFileName: 目标文件

    Returns:
        None

    '''
    with open(label_file, 'w') as f:
        json.dump(labels, f)
        print('write labels to %s, num is %d' % (label_file, len(labels)))

    return


def copy_all_files(src_dir, dst_dir):
    '''
    拷贝源目录下所有文件至目标目录
    Args:
        src_dir:
        dst_dir:

    Returns:

    '''
    pics_augmetation.ensure_dir(src_dir, flag=1)
    pics_augmetation.ensure_dir(dst_dir, flag=1)

    dir_names = os.listdir(src_dir)
    for pic_dir in dir_names:
        if os.path.isdir(src_dir + str(pic_dir)):
            for name in os.listdir(src_dir + str(pic_dir)):
                shutil.copy(src_dir + str(pic_dir) + '/' + name, dst_dir)

    print("From %s To %s" % (src_dir, dst_dir))
    print("拷贝完成")


# 根据比例分配labels文件成train、val、test用的labels
def prepare_dataset(label_file, src_dir,
                    train_label='train.json', val_label='val.json',
                    train_dir=TRAIN_DIR, val_dir=VAL_DIR, ratio=TRAIN_RATIO):
    '''
    准备数据集文件
    根据全标签文件，按比例分为训练集和验证集
    生成对应的标签文件
    并拷贝对应文件至目的文件夹

    Args:
        label_file:
        src_dir:
        train_label:
        val_label:
        train_dir:
        val_dir:
        ratio:

    Returns:

    '''
    data_dict = {}
    # 读入所有标签
    with open(label_file, 'r') as f:
        label_list = json.load(f)
        # shuffle
        random.shuffle(label_list)

    # 建立文件-标签字典
    for image in label_list:
        data_dict[image['image_id']] = int(image['label_id'])

    pics_augmetation.ensure_dir(train_dir, flag=1)
    pics_augmetation.ensure_dir(val_dir, flag=1)

    # 按比例分配训练集和验证集文件
    print(len(data_dict))
    img_list = list(data_dict.keys())
    # print(img_list)
    total_nums = len(img_list)
    print(total_nums)
    train = img_list[: math.floor(ratio * total_nums)]
    val = img_list[math.floor(ratio * total_nums):]
    print("train: %d, val: %d" % (len(train), len(val)))

    # 建立训练集标签文件，拷贝对应文件至目标目录
    train_labels = []
    # print(train)
    for item in train:
        temp_dict = {}
        temp_dict['label_id'] = int(data_dict[item])
        temp_dict['image_id'] = item
        train_labels.append(temp_dict)
        shutil.copy(src_dir + item, TRAIN_DIR)
    save_labels(train_labels, 'train.json')

    # 建立验证集标签文件，拷贝对应文件至目标目录
    val_labels = []
    for item in val:
        temp_dict = {}
        temp_dict['label_id'] = int(data_dict[item])
        temp_dict['image_id'] = item
        val_labels.append(temp_dict)
        shutil.copy(src_dir + item, VAL_DIR)
    save_labels(val_labels, 'val.json')

    return


if __name__ == '__main__':
    ori_pic_dir = './testdstpic/'
    all_pic_dir = './allPics/'
    label_path = ENTIRE_LABELFILE

    # 将按目录存放的文件录入成标签信息
    entire_label = files2label(ori_pic_dir)

    # 生成全标签文件
    save_labels(entire_label, label_path)

    # 汇总所有文件
    copy_all_files(ori_pic_dir, all_pic_dir)

    # 准备数据集 生成训练集和验证集的目录及标签文件
    prepare_dataset(label_path, all_pic_dir)

    print("===Finish===")