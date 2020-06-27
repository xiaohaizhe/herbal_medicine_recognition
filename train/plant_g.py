
import tensorflow as tf
import numpy as np
import time
import argparse
import shutil
import json
import os
import herb_input
import inception_v3 as v3
import tensorflow.contrib.slim as slim
from datetime import datetime

BATCH_SIZE = 8
IMAGE_SIZE = 128
IMAGE_GNET_SIZE = 299
IMAGE_CHANNEL = 3
CLASS_NUMBER = 38
CHECKFILE = './checkpoint/model.ckpt'
MODEL_DIR = './checkpoint/'
BESTMODEL_DIR = './bestmodel/'
LOGNAME = 'herb'
LEARNINGRATE = 1e-2
VALLI_OPEN = 1

VAL_DIR = './val/'
VAL_ANNO = './val.json'
VAL_OPEN = 1

# 定义input_images为图片数据
input_images = tf.placeholder(dtype=tf.float32, shape=[None, IMAGE_GNET_SIZE, IMAGE_GNET_SIZE, IMAGE_CHANNEL], name='input')
# 定义input_labels为labels数据
input_labels = tf.placeholder(dtype=tf.float32, shape=[None], name="label")
one_hot_labels = tf.one_hot(indices=tf.cast(input_labels, tf.int32), depth=CLASS_NUMBER)

# 定义dropout的概率
keep_prob = tf.placeholder(tf.float32, name='keep_prob')
is_training = tf.placeholder(tf.bool, name='is_training')

def vali_evaluation(sess, data, loss, accuracy, val_nums):
    val_max_steps = int(val_nums / BATCH_SIZE)
    val_losses = []
    val_accs = []
    for _ in range(val_max_steps):
        val_x, val_y = data.next_batch(BATCH_SIZE, IMAGE_GNET_SIZE)
        # print('labels:',val_y)
        # val_loss = sess.run(loss, feed_dict={x: val_x, y: val_y, keep_prob: 1.0})
        # val_acc = sess.run(accuracy,feed_dict={x: val_x, y: val_y, keep_prob: 1.0})
        val_loss, val_acc = sess.run([loss, accuracy], feed_dict={input_images: val_x, input_labels: val_y, keep_prob:1.0, is_training: False})
        val_losses.append(val_loss)
        val_accs.append(val_acc)
    # mean_loss = np.array(val_losses, dtype=np.float32).mean()
    mean_acc = np.array(val_accs, dtype=np.float32).mean()
    return mean_loss, mean_acc

def train(train_dir, annotations, max_step, checkpoint_dir='./checkpoint/'):
    max_acc = 0.0
    # train the model
    plant_data = herb_input.plant_data_fn(train_dir, annotations)
    val_data = herb_input.plant_data_fn(VAL_DIR, VAL_ANNO)
    val_size = val_data.data_counts()

    # Define the model:
    with slim.arg_scope(v3.inception_v3_arg_scope()):
        out, end_points = v3.inception_v3(inputs=input_images, num_classes=CLASS_NUMBER,
                                            dropout_keep_prob=keep_prob, is_training=is_training)

    # Specify the loss function: tf.losses定义的loss函数都会自动添加到loss函数,不需要add_loss()了
    tf.losses.softmax_cross_entropy(onehot_labels=one_hot_labels, logits=out)#添加交叉熵损失
    # slim.losses.add_loss(my_loss)
    loss = tf.losses.get_total_loss(add_regularization_losses=True)#添加正则化损失
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(out, 1), tf.argmax(one_hot_labels, 1)), tf.float32))

    # define the tensorboard summary info
    try:
        image_summary = tf.image_summary
        scalar_summary = tf.scalar_summary
        histogram_summary = tf.histogram_summary
        merge_summary = tf.merge_summary
        SummaryWriter = tf.train.SummaryWriter
    except:
        image_summary = tf.summary.image
        scalar_summary = tf.summary.scalar
        histogram_summary = tf.summary.histogram
        merge_summary = tf.summary.merge
        SummaryWriter = tf.summary.FileWriter

    # Specify the optimization scheme:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=LEARNINGRATE)

    # 在定义训练的时候, 注意到我们使用了`batch_norm`层时,需要更新每一层的`average`和`variance`参数,
    # 更新的过程不包含在正常的训练过程中, 需要我们去手动像下面这样更新
    # 通过`tf.get_collection`获得所有需要更新的`op`
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    # 使用`tensorflow`的控制流, 先执行更新算子, 再执行训练
    with tf.control_dependencies(update_ops):
        # create_train_op that ensures that when we evaluate it to get the loss,
        # the update_ops are done and the gradient updates are computed.
        # train_op = slim.learning.create_train_op(total_loss=loss,optimizer=optimizer)
        train_op = slim.learning.create_train_op(total_loss=loss, optimizer=optimizer)

    with tf.Session() as sess:
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            print('Restore the model from checkpoint %s' % ckpt.model_checkpoint_path)
            # Restores from checkpoint
            saver.restore(sess, ckpt.model_checkpoint_path)
            start_step = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
        else:
            sess.run(tf.global_variables_initializer())
            start_step = 0
            print('start training from new state')
        logger = herb_input.train_log(LOGNAME)

        for step in range(start_step, start_step + max_step):
            start_time = time.time()
            x, y = plant_data.next_batch(BATCH_SIZE, IMAGE_GNET_SIZE)
            _, train_loss = sess.run([train_op, loss], feed_dict={input_images: x, input_labels: y,
                                                                    keep_prob: 0.5, is_training: True})
            if step % 50 == 0:
                #outputs = sess.run(out, feed_dict={input_images: x, input_labels: y,
                #                                                keep_prob: 1.0, is_training: False})
                train_accuracy = sess.run(accuracy, feed_dict={input_images: x, input_labels: y,
                                                                keep_prob: 1.0, is_training: False})
                #train_loss = sess.run(cross_entropy, feed_dict={features: x, labels: y, keep_prob: 1})

                # summary
                loss_summary = scalar_summary('loss', train_loss)
                acc_summary = scalar_summary('accuracy', train_accuracy)
                merged = merge_summary([loss_summary, acc_summary])
                summary = sess.run(merged, feed_dict={features: x, labels: y, keep_prob: 1})
                writer.add_summary(summary, step)

                duration = time.time() - start_time
                logger.info("step %d: training accuracy %g, loss is %g (%0.3f sec)" % (step, train_accuracy, train_loss, duration))
                #print(outputs)
            if step % 1000 == 1:
                saver.save(sess, CHECKFILE, global_step=step)
                print('writing checkpoint at step %s' % step)

            if VAL_OPEN and step % 5000 == 1:
                # 验证准确率
                print('step into validation, data_size:%d' % val_size)
                mean_loss, mean_acc = vali_evaluation(sess, val_data, loss, accuracy, val_size)
                print("%s: Step [%d]  val Loss : %f, val accuracy :  %g" % (datetime.now(), step, mean_loss, mean_acc))

                if mean_acc > max_acc and mean_acc > 0.7:
                    # 保存val准确率最高的模型
                    max_acc = mean_acc
                    best_models = os.path.join(MODEL_DIR, 'best_models_{:.4f}.ckpt'.format(max_acc))
                    print('{}------save:{}'.format(datetime.now(), best_models))
                    saver.save(sess, best_models)
                    # 拷贝最佳模型到指定目录
                    if BESTMODEL_DIR:
                        if not os.path.exists(BESTMODEL_DIR):
                            os.makedirs(BESTMODEL_DIR)
                        shutil.copy(best_models+'.meta', BESTMODEL_DIR)
                        shutil.copy(best_models+'.index', BESTMODEL_DIR)
                        shutil.copy(best_models+'.data-00000-of-00001', BESTMODEL_DIR)


def test(test_dir, checkpoint_dir='./checkpoint/'):
    # predict the result 
    test_images = os.listdir(test_dir)

    '''new'''
    # Define the model:
    with slim.arg_scope(v3.inception_v3_arg_scope()):
        out, end_points = v3.inception_v3(inputs=input_images, num_classes=CLASS_NUMBER,
                                            dropout_keep_prob=1.0, is_training=False)

    score = tf.nn.softmax(out, name='pre')
    class_id = tf.argmax(score, 1)
    '''new'''

    with tf.Session() as sess:
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            print('Restore the model from checkpoint %s' % ckpt.model_checkpoint_path)
            # Restores from checkpoint
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            raise Exception('no checkpoint find')

        result = []
        for test_image in test_images:
            temp_dict = {}
            x = herb_input.img_resize(os.path.join(test_dir, test_image), IMAGE_GNET_SIZE)
            pre_label = sess.run(class_id, feed_dict={input_images:np.expand_dims(x, axis=0), keep_prob: 1})
            temp_dict['image_id'] = test_image
            temp_dict['label_id'] = pre_label.tolist()
            result.append(temp_dict)
            print('image %s is %d' % (test_image, pre_label[0]))

        with open('submit.json', 'w') as f:
            json.dump(result, f)
            print('write result json, num is %d' % len(result))

def val(val_dir, checkpoint_dir='./checkpoint/'):
    # predict the result 
    val_images = os.listdir(val_dir)

    # Define the model:
    with slim.arg_scope(v3.inception_v3_arg_scope()):
        out, end_points = v3.inception_v3(inputs=input_images, num_classes=CLASS_NUMBER,
                                            dropout_keep_prob=1.0, is_training=False)

    score = tf.nn.softmax(out, name='pre')
    class_id = tf.argmax(score, 1)

    with tf.Session() as sess:
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            print('Restore the model from checkpoint %s' % ckpt.model_checkpoint_path)
            # Restores from checkpoint
            saver.restore(sess, ckpt.model_checkpoint_path)
            start_step = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
        else:
            raise Exception('no checkpoint find')

        result = []
        loop = 0
        with open('val_pre.json', 'w') as f:
            for val_image in val_images:
                temp_dict = {}
                x = herb_input.img_resize(os.path.join(val_dir, val_image), IMAGE_GNET_SIZE)
                pre_label = sess.run(class_id, feed_dict={input_images:np.expand_dims(x, axis=0), keep_prob: 1})
                temp_dict['image_id'] = val_image
                temp_dict['label_id'] = pre_label.tolist()
                result.append(temp_dict)
                loop = loop+1
                print('image %s is %d' % (val_image, pre_label[0]))

                if loop % 1000 == 1:
                    json.dump(result, f)
                    print('loop:%d, write result json, num is %d' % (loop, len(result)))
                    del result[:]

            json.dump(result, f)
            print('loop end %d, write result json, num is %d' % (loop, len(result)))


def calWholeAcc(train_dir, annotations, checkpoint_dir='./checkpoint/'):
    # calculate train accuracy
    plant_data = herb_input.plant_data_fn(train_dir, annotations)
    x, y = plant_data.whole_batch(IMAGE_GNET_SIZE)

    # Define the model:
    with slim.arg_scope(v3.inception_v3_arg_scope()):
        out, end_points = v3.inception_v3(inputs=input_images, num_classes=CLASS_NUMBER,
                                            dropout_keep_prob=keep_prob, is_training=is_training)

    # Specify the loss function: tf.losses定义的loss函数都会自动添加到loss函数,不需要add_loss()了
    tf.losses.softmax_cross_entropy(onehot_labels=one_hot_labels, logits=out)#添加交叉熵损失
    # slim.losses.add_loss(my_loss)
    loss = tf.losses.get_total_loss(add_regularization_losses=True)#添加正则化损失
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(out, 1), tf.argmax(one_hot_labels, 1)), tf.float32))

    # Specify the optimization scheme:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=LEARNINGRATE)

    # 在定义训练的时候, 注意到我们使用了`batch_norm`层时,需要更新每一层的`average`和`variance`参数,
    # 更新的过程不包含在正常的训练过程中, 需要我们去手动像下面这样更新
    # 通过`tf.get_collection`获得所有需要更新的`op`
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    # 使用`tensorflow`的控制流, 先执行更新算子, 再执行训练
    with tf.control_dependencies(update_ops):
        # create_train_op that ensures that when we evaluate it to get the loss,
        # the update_ops are done and the gradient updates are computed.
        # train_op = slim.learning.create_train_op(total_loss=loss,optimizer=optimizer)
        train_op = slim.learning.create_train_op(total_loss=loss, optimizer=optimizer)

    with tf.Session() as sess:
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            print('Restore the model from checkpoint %s' % ckpt.model_checkpoint_path)
            # Restores from checkpoint
            saver.restore(sess, ckpt.model_checkpoint_path)
            start_step = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
        else:
            sess.run(tf.global_variables_initializer())
            start_step = 0
            print('start training from new state')
        logger = herb_input.train_log(LOGNAME)

        start_time = time.time()
        train_accuracy = sess.run(accuracy, feed_dict={input_images: x, input_labels: y,
                                                        keep_prob: 1.0, is_training: False})
        duration = time.time() - start_time
        logger.info("whole training accuracy %g(%0.3f sec)" % (train_accuracy, duration))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--mode',
        type=str,
        default='train',
        help="""\
        determine train or test\
        """
    )

    parser.add_argument(
        '--train_dir',
        type=str,
        default='./train_pics/',
        help="""\
        determine path of trian images\
        """
    )

    parser.add_argument(
        '--annotations',
        type=str,
        default='./train.json',
        help="""\
        annotations for train images\
        """
    )
    parser.add_argument(
        '--test_dir',
        type=str,
        # default='../ai_challenger_plant_validation_20170908/plant_validation_images_20170908/',
        default='../plant_disease/11.14/ai_challenger_pdr2018_validationset_20181023/AgriculturalDisease_validationset/images',
        help="""\
        determine path of test images\
        """
    )
    parser.add_argument(
        '--max_step',
        type=int,
        default=65000,
        help="""\
        determine maximum training step\
        """
    )

    FLAGS = parser.parse_args()
    print("##############")

    if FLAGS.mode == 'train':
        print("#####START TRAIN#####")
        print(FLAGS.train_dir)
        print(FLAGS.annotations)
        print(FLAGS.max_step)
        train(FLAGS.train_dir, FLAGS.annotations, FLAGS.max_step)
    elif FLAGS.mode == 'test':
        print("#####START TEST#####")
        print(FLAGS.test_dir)
        start_time = time.time()
        test(FLAGS.test_dir)
        duration = time.time() - start_time
        print("use time : %0.3f sec" % duration)


    else:
        raise Exception('error mode')
print('done')
