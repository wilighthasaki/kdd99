import tensorflow as tf
import numpy as np
import argparse
import os
from CNNClassifier import CNNClassifier
from preprocess import Preproccessor

# 使用方式 python3 train.py train_features train_y model_path

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data_path', help='the path of the train data file', default='./data/kddcup.data_10_percent')
parser.add_argument('-p', '--preprocess_path', help='the path to save preprocess model', default='./output/models/preprocess.pkl')
parser.add_argument('-m', '--model_path', help='the path to save model', default='./output/models/model')
parser.add_argument('-g', '--gpu_config', help='the config of gpu, the default is /cpu:0', default='/cpu:0')

args = parser.parse_args()

# 获取输入的特征，输出，和保存模型的路径，在这里我们采用之前预处理过的特征，
# 如果有需要也可以修改成为在这里进行预处理，只要在这里初始化一个预处理类即可

print('正在预处理数据')
pre = Preproccessor(args.data_path)
features = pre.get_features()
y = pre.get_y()
pre.save_model(args.preprocess_path)

model_path = args.model_path
gpu_config = args.gpu_config

# 创建输出路径
if not os.path.exists('./output'):
    os.mkdir('./output')
if not os.path.exists('./output/models/'):
    os.mkdir('./output/models/')

# 划分训练集和验证集
data_len = features.shape[0]
shuffle_idx = np.arange(data_len)
np.random.shuffle(shuffle_idx)
train_features = features[shuffle_idx[:int(data_len * 4 / 5)]]
train_y = y[shuffle_idx[:int(data_len * 4 / 5)]]
val_features = features[shuffle_idx[int(data_len * 4 / 5):]]
val_y = y[shuffle_idx[int(data_len * 4 / 5):]]

print('正在初始化模型')
# 这个设置用来使没有GPU的时候可以调用CPU
config = tf.ConfigProto(allow_soft_placement=True)
with tf.Session(config=config) as sess:
    with tf.device(gpu_config):
        # 指定网络参数初始化方式
        initializer = tf.random_uniform_initializer(-0.1, 0.1)

        model = CNNClassifier(train_features.shape[1], train_y.shape[1])

        print('开始训练')
        tf.global_variables_initializer().run()
        model.train(sess, train_features, train_y, val_features, val_y, num_epochs=2, model_path=model_path)




