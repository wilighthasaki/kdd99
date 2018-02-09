import tensorflow as tf
from get_batches import get_batches

class CNNClassifier(object):
    '''
    这是用CNN实现的分类器，CNN结构在另外一个文件中介绍。
    '''
    def __init__(self, input_size, num_classes, is_train=True):

        # 输入输出
        self.image_input = tf.placeholder(tf.float32, [None, input_size])
        self.y_ = tf.placeholder(tf.int64, [None, num_classes])

        # 整理成 6*6的格式作为CNN的输入
        self.network_input = tf.layers.dense(self.image_input, 36, name='input_dense')
        self.network_input = tf.reshape(self.network_input, [-1, 6, 6, 1])

        # 网络结构
        with tf.variable_scope('model'):
            with tf.variable_scope('conv1'):
                # 卷积层1参数
                self.conv1_weights = tf.get_variable(
                    'conv1_w',
                    shape=[3, 3, 1, 32],
                    dtype=tf.float32
                )
                self.conv1_biases = tf.get_variable(
                    'conv1_b',
                    shape=[32]
                )
                # 卷积层1
                self.conv1 = tf.nn.conv2d(self.network_input,
                                          self.conv1_weights,
                                          strides=[1, 1, 1, 1],
                                          padding='SAME')
                self.relu1 = tf.nn.relu(tf.nn.bias_add(self.conv1, self.conv1_biases))

            with tf.variable_scope('conv2'):
                # 卷积层2参数
                self.conv2_weights = tf.get_variable(
                    'conv2_w',
                    shape=[3, 3, 32, 64],
                    dtype=tf.float32
                )
                self.conv2_biases = tf.get_variable(
                    'conv2_b',
                    shape=[64]
                )
                # 卷积层2
                self.conv2 = tf.nn.conv2d(self.relu1,
                                          self.conv2_weights,
                                          strides=[1, 1, 1, 1],
                                          padding='SAME')
                self.relu2 = tf.nn.relu(tf.nn.bias_add(self.conv2, self.conv2_biases))

                # 池化层
                self.pool = tf.nn.max_pool(self.relu2,
                                           ksize=[1, 2, 2, 1],
                                           strides=[1, 2, 2, 1],
                                           padding='SAME')

            # 全连接层1
            with tf.variable_scope('dense1'):
                pool_shape = self.pool.get_shape().as_list()

                # 全连接层1参数
                self.fc1_weights = tf.get_variable('fc1_w',
                                                   shape=[pool_shape[1] * pool_shape[2] * pool_shape[3], 512])
                self.fc1_biases = tf.get_variable('fc1_b',
                                                  shape=[512])

                # 将卷基层的特征整理成一列向量，再经过全连接层
                self.reshape = tf.reshape(self.pool,
                                          shape=[-1, pool_shape[1] * pool_shape[2] * pool_shape[3]])
                self.hidden = tf.nn.relu(
                        tf.matmul(self.reshape, self.fc1_weights) + self.fc1_biases)

            with tf.variable_scope('dense2'):
                # if is_train:
                #     self.hidden = tf.nn.dropout(self.hidden, 0.5)
                self.fc2_weights = tf.get_variable('fc2_w',
                                                   shape=[512, num_classes])
                self.fc2_biases = tf.get_variable('fc2_b',
                                                  shape=[num_classes])
                self.y = tf.identity(
                        tf.matmul(self.hidden, self.fc2_weights) + self.fc2_biases)

            with tf.variable_scope('classify'):
                # 预测出的分类
                self.y_op = tf.argmax(self.y, axis=1)

                # 计算出的loss值
                self.loss = tf.nn.softmax_cross_entropy_with_logits(labels=self.y_, logits=self.y)

                # 准确率
                self.acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.y, axis=1), tf.argmax(self.y_, axis=1)), tf.float32))

                # 使用Adam优化器来进行梯度下降
                self.train_op = tf.train.AdamOptimizer(learning_rate=0.001,
                                                       epsilon=1e-08,
                                                       use_locking=False).minimize(self.loss)

    def train(self, sess, x_train, y_train, x_val, y_val, num_epochs=10, model_path='models/model1'):
        for epoch in range(num_epochs):
            # 训练集
            print('epoch: %d' % epoch)
            train_loss = []
            train_acc = []
            x_batches, y_batches = get_batches([x_train, y_train], batch_size=32, shuffle=True)
            for x, y in zip(x_batches, y_batches):
                loss, _, acc, tar = sess.run(
                    [self.loss,
                     self.train_op,
                     self.acc,
                     self.y],
                    feed_dict={
                        self.image_input: x,
                        self.y_: y
                    }
                )
                train_loss += list(loss)
                train_acc.append(acc)
            print('training loss: %.7f' % (sum(train_loss)/len(train_loss)))
            print('training acc: %.7f' % (sum(train_acc)/len(train_acc)))
            # 验证集
            val_loss = []
            val_acc = []
            x_batches, y_batches = get_batches([x_val, y_val], batch_size=32, shuffle=True)
            for x, y in zip(x_batches, y_batches):
                loss, acc = sess.run(
                    [self.loss,
                     self.acc],
                    feed_dict={
                        self.image_input:x,
                        self.y_: y
                    }
                )
                val_loss += list(loss)
                val_acc.append(acc)
            print('validation loss: %.7f' % (sum(val_loss)/len(val_loss)))
            print('validation acc: %.7f' % (sum(val_acc)/len(val_acc)))
            saver = tf.train.Saver()
            saver.save(sess, model_path)

    def predict(self, sess, x_input):
        target = sess.run(
            self.y_op,
            feed_dict={
                self.image_input: x_input
            }
        )
        return target
