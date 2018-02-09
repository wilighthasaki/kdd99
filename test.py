from preprocess import Preproccessor
import numpy as np
import tensorflow as tf
from CNNClassifier import CNNClassifier
from evaluate import evaluate

model_path = './output/models/model'
gpu_config = '/cpu:0'

print('正在预处理测试集')
pre = Preproccessor('./data/corrected', './output/models/preprocess.pkl')

test_y = pre.get_y()
test_label = np.argmax(test_y, axis=1)
test_features = pre.get_features()

print('正在初始化模型')
# 这个设置用来使没有GPU的时候可以调用CPU
config = tf.ConfigProto(allow_soft_placement=True)
with tf.Session(config=config) as sess:
    with tf.device(gpu_config):
        # 指定网络参数初始化方式
        model = CNNClassifier(test_features.shape[1], test_y.shape[1])
        saver = tf.train.Saver()
        saver.restore(sess, model_path)
        print('开始测试')
        results = []
        for i in test_features:
            results.append(model.predict(sess, [i])[0])
        confusion_matrix, acc = evaluate(results, test_label)
        print('confusion matrix')
        print(confusion_matrix.round(4))

        print('accuracy:')
        print(acc)




