# -*- coding:utf-8 -*-

# 中草药种类识别

# 初始化工作
from flask import request, Flask
import json
import os
import uuid

import numpy as np
import inception_v3 as v3
import tensorflow as tf
import tensorflow.contrib.slim as slim
import herb_utils


# 全局
app = Flask(__name__)
FLASK_APP_HOST = '0.0.0.0'                                          # 服务启动地址
FLASK_APP_PORT = 5055                                               # APP服务启动端口
FLASK_APP_DEBUG = False                                             # 服务调试模式
IMAGE_GNET_SIZE = 299                                               # 图片格式化尺寸
CLASS_NUMBER = 9                                                    # 图片分类个数
LOG_NAME = 'herb'                                                   # 日志文件前缀
MODEL_FLIE = './tmmodel/'                                           # 训练模型路径
META_FILE = './tmmodel/modelTM-999.meta'                            # 图文件路径
LABEL_FILE = './labels_cn_tianma.txt'                               # 中文标签文件路径
LABEL_FILE_EN = './labels_en_tianma.txt'                            # 英文标签文件路径
UPLOAD_DIR = './upload/'                                            # 图片路径

# 创建logger
logger = herb_utils.train_log(LOG_NAME)

# 载入标签
labels = np.loadtxt(LABEL_FILE, str, delimiter='\t')
labels_en = np.loadtxt(LABEL_FILE_EN, str, delimiter='\t')

# 定义input_images为图片数据
input_images = tf.placeholder(dtype=tf.float32, shape=[None, IMAGE_GNET_SIZE, IMAGE_GNET_SIZE, 3], name='input')
# 定义dropout的概率
keep_prob = tf.placeholder(tf.float32, name='keep_prob')

# Define the model:
print("Define the model--------------------")
with slim.arg_scope(v3.inception_v3_arg_scope()):
    out, end_points = v3.inception_v3(inputs=input_images, num_classes=CLASS_NUMBER,
                                        dropout_keep_prob=1.0, is_training=False)

sess = 0
scores = tf.nn.softmax(out, name='pre')
values, indices = tf.nn.top_k(scores, 3)


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
    global sess
    result = []
    code = 0

    x = herb_utils.img_resize(image_file, IMAGE_GNET_SIZE)
    if sess:
        pre_score, pre_label = sess.run([values, indices], feed_dict={input_images: np.expand_dims(x, axis=0), keep_prob: 1})
        for i in range(3):
            # 输出概率最高的3个分类
            temp_dict = {}
            index = int(pre_label[0][i])
            score = format(pre_score[0][i], '.2%')
            if score != "0.00%":
                # 不输出概率为0的分类
                temp_dict['label_id'] = str(index)
                temp_dict['class'] = str(labels_en[index])
                temp_dict['name'] = str(labels[index])
                temp_dict['score'] = score
                result.append(temp_dict)
    else:
        logger.error("session error!")
        code = -1
        result = 'predict error'

    return code, result


# 测试
@app.route('/')
def hello():
    return "HELLO!"


# 图片分类预测接口
@app.route('/app/upload', methods=['POST'])
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
    logger.info("%s result:code[%s]" % (f_name, response['result']))
    #logger.info("%s result:code" % {f_name})
    logger.debug("------upload method exit------")    
    return json.dumps(response, ensure_ascii=False)


# 启动服务
if __name__ == '__main__':
    logger.info("===== server start =====")
    herb_utils.check_img_dir(UPLOAD_DIR)
    create_sess()
    app.run(host=FLASK_APP_HOST, port=FLASK_APP_PORT, debug=FLASK_APP_DEBUG)
