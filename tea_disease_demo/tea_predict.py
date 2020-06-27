# -*- coding:utf-8 -*-

# 茶叶病虫害识别
"""
当前识别类型：{
        01:茶白星病;02:茶赤叶斑病;03:茶胴枯病;04:茶褐色叶斑病;
        05:茶苗白绢病;06:茶树炭疽病;07:;08:茶叶轮斑病;
        09:茶叶叶枯病;10:茶圆赤星病;11:茶藻斑病
}
类型编号和实际名称以label文件为准
"""


# 初始化工作
from flask import request, Flask
import json
import os
import uuid

import numpy as np
#import inception_v3 as v3
#import tensorflow as tf
#import tensorflow.contrib.slim as slim
import predict_utils


# 全局
app = Flask(__name__)
FLASK_APP_HOST = '127.0.0.1'                                        # 服务启动地址
FLASK_APP_PORT = 5000                                               # APP服务启动端口
FLASK_APP_DEBUG = False                                             # 服务调试模式
IMAGE_GNET_SIZE = 299                                               # 图片格式化尺寸
CLASS_NUMBER = 11                                                   # 图片分类个数
LOG_NAME = 'tea_disease'                                            # 日志文件前缀
MODEL_FLIE = './models/'                                            # 训练模型路径
META_FILE = './model/model.ckpt-new.meta'                           # 图文件路径
LABEL_FILE = './label_cn.txt'                                       # 标签文件路径
UPLOAD_DIR = './upload/'                                            # 图片路径

# 创建logger
logger = predict_utils.train_log(LOG_NAME)

# 载入标签
labels = np.loadtxt(LABEL_FILE, str, delimiter='\t')
'''
# 定义input_images为图片数据
input_images = tf.placeholder(dtype=tf.float32, shape=[None, IMAGE_GNET_SIZE, IMAGE_GNET_SIZE, 3], name='input')
# 定义dropout的概率
keep_prob = tf.placeholder(tf.float32, name='keep_prob')

# Define the model:
print("Define the model--------------------")
with slim.arg_scope(v3.inception_v3_arg_scope()):
    out, end_points = v3.inception_v3(inputs=input_images, num_classes=CLASS_NUMBER,
                                        dropout_keep_prob=1.0, is_training=False)


scores = tf.nn.softmax(out, name='pre')
values, indices = tf.nn.top_k(scores, 3)
'''
sess = 0


# 载入模型创建Sesssion
def create_sess():
    global sess
    sess = tf.Session()
    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state(MODEL_FLIE)
    if ckpt and ckpt.model_checkpoint_path:
        logger.info('Restore the model from checkpoint %s' % ckpt.model_checkpoint_path)
        # Restores from checkpoint
        saver.restore(sess, ckpt.model_checkpoint_path)
        logger.info('Restore model SUCCESS!')
    else:
        logger.error('No model checkpoint find')


# 预测分类
def predict(image_file):
    """
    预测茶树病虫害类型
    code : 结果码 --> int
    result : 图片预测结果 --> list
    """
    global sess
    result = []
    code = 0

    # 检查图片
    img = predict_utils.img_resize(image_file, -1)
    picIndex = predict_utils.get_exif_value(img, 'ExifImageHeight')
    logger.info('picture index: %d' % picIndex)

    # 读取图片所属类别
    indexList = predict_utils.get_pic_index(picIndex)

    # 生成概率
    scores = predict_utils.get_pic_prob()

    # 生成其他两个标签
    # result:输出3个分类
    # code: 结果码

    if code == 0:
        for i in range(3):
            # 输出概率最高的3个分类
            temp_dict = {}
            if scores[i] != 0:
                # 不输出概率为0的分类
                temp_dict['label_id'] = str(indexList[i])
                temp_dict['name'] = str(labels[indexList[i]])
                temp_dict['score'] = format(scores[i] / 10000, '.2%')
                result.append(temp_dict)
    else:
        logger.error("picture error!")
        code = -1
        result = 'picture error'

    return code, result


# 测试
@app.route('/')
def hello():
    return "HELLO!"


# 图片分类预测接口
@app.route('/upload', methods=['POST'])
def upload():
    logger.debug("------upload method entry------")
    response = {}

    # 解析数据
    try:
        file = request.files['file']
    except:
        # 无文件
        logger.error("get post file ERROR!") 
        response['code'] = -2
        response['result'] = "missing img file"
        return json.dumps(response)

    logger.debug("get post file:%s", {file}) 
    extension = os.path.splitext(file.filename)[1]
    f_name = str(uuid.uuid4()) + extension
    file.save(UPLOAD_DIR + f_name)
    response['code'], response['result'] = predict(UPLOAD_DIR + f_name)
    logger.debug("------upload method exit------")    
    return json.dumps(response, ensure_ascii=False)


# 启动服务
if __name__ == '__main__':
    logger.info("===== server start =====")
    predict_utils.check_img_dir(UPLOAD_DIR)
    # create_sess()
    app.run(host=FLASK_APP_HOST, port=FLASK_APP_PORT, debug=FLASK_APP_DEBUG)
