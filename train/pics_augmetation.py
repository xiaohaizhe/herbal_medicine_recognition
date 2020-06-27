# coding :utf-8
# 对目标目录下的图片进行扩展
# 通过将图片上下翻转、左右翻转、
# 逆时针旋转90度、180度、270度
# 形成6倍的图片数据文件

from PIL import Image
import os

def ensure_dir(dirPath, flag=0):
    '''
    检查目录路径
    Args:
        dirPath: 目录路径
        flag: 是否创建，1：创建；其他：退出

    Returns:
        None
    '''
    isExists = os.path.exists(dirPath)
    if not isExists:
        print("不存在目录:" + str(dirPath), end="")
        if flag == 1:
            print(" 正在生成...")
            os.makedirs(dirPath)
            return
        else :
            print(" 退出！")
            return


def augment_pics(srcDir, dstDir):
    '''
    扩充图片数量
    将源图片目录中的所有图片通过翻转和选装，扩展成6倍的图片至目标目录
    Args:
        srcDir: 图片源目录
        dstDir: 图片目标目录

    Returns:
        None
    '''
    ensure_dir(srcDir)
    ensure_dir(dstDir, flag = 1)

    files = os.listdir(srcDir)
    i = 0
    print("目录下有文件" + str(len(files)) + "个")
    for file in files:
        if str(file).startswith('.'):
            # 跳过隐藏文件
            continue

        with Image.open(srcDir + file) as im:
            i += 1
            # 左右翻转，上下翻转，逆时针旋转90， 180， 270度
            im1 = im.transpose(Image.FLIP_LEFT_RIGHT)
            im2 = im.transpose(Image.FLIP_TOP_BOTTOM)
            im3 = im.transpose(Image.ROTATE_90)
            im4 = im.transpose(Image.ROTATE_180)
            im5 = im.transpose(Image.ROTATE_270)

            # 提取文件名，分隔文件名和后缀
            name1, suffix = file.split('.')
            # print(name1, suffix)

            # 返回工作目录 防止相对路径问题
            # os.chdir(workingPath)

            # 保存6种图片至指定目录
            im.save(dstDir + str(name1) + '_ORI.' + str(suffix))
            im1.save(dstDir + str(name1) + '_LR.' + str(suffix))
            im2.save(dstDir + str(name1) + '_TB.' + str(suffix))
            im3.save(dstDir + str(name1) + '_LEFT.' + str(suffix))
            im4.save(dstDir + str(name1) + '_DOWN.' + str(suffix))
            im5.save(dstDir + str(name1) + '_RIGHT.' + str(suffix))

        if (i % 100 == 1):
            print(str(i) + "->", end='')

    print("finish!")
    print("FROM " + str(srcDir))
    print("TO " + str(dstDir))
    print(str(i) + "pics completed!")
    return


if __name__ == '__main__':
    picSrcDir = "./herb_pic_ori/"
    picDstDir = "./newPicDir/"

    srcDirs = os.listdir(picSrcDir)

    labels = {"白花蛇舌草": 0, "白芍": 1, "白术": 2, "苍术": 3, "柴胡": 4, \
              "川芎": 5, "丹参": 6, "党参": 7, "甘草": 8, "红花": 9, "黄连": 10, \
              "黄芪": 11, "菊花": 12, "山药": 13, "生地黄": 14, "太子参": 15, "天麻": 16, \
              "仙鹤草": 17, "续断": 18}

    i = 0
    for item in srcDirs:
        if os.path.isdir(picSrcDir + item) and (item in labels):
            dstD = str(picDstDir) + str(labels[item]) + '/'
            srcD = str(picSrcDir) + str(item) + '/'
            # print("src: " + str(srcD))
            # print("dst: " + str(dstD))
            augment_pics(srcD, dstD)
            i += 1

    print("legal dir nums:" + str(i))
