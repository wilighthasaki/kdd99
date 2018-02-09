'''
这是对初始CSV文件进行预处理的代码，主要是对数据进行读取，对标量进行one-hot编码和对数值类型的特征进行归一化。
'''

import numpy as np
import pandas as pd
import pickle
import os


class Preproccessor(object):
    '''
    预处理的类，初始化时数据的输入和输出路径
    '''
    def __init__(self, input_path, model_path=None):
        # 读取数据
        data = pd.read_csv(input_path, names=[i for i in range(42)])
        # data = data.loc[:10, :]

        # 获取攻击类型的转换表
        class_map = self.get_class_map()
        self.features = []

        # 对于训练集的处理，处理后将对应的模型保存
        if not model_path:
            # 用来保存预处理中的各个模型，包括每一维的one hot或归一化参数.
            self.model_map = {}

            # 对输出的标签进行onehot处理
            input_y = [tag.rstrip().rstrip('.') for tag in data[41]]
            input_y = [class_map[i] for i in input_y]
            # 使用one_hot编码，获得处理过的one_hot编码和对应的转化字典
            self.y_one_hot, y_dict = self.one_hot_encoding(input_y)
            self.model_map['y'] = y_dict

            # 对数据集中的离散数据进行one_hot编码，同时返回字典。
            # 这里存在one_hot编码过于稀疏的问题，后续可能可以采用其他方法代替
            # 数据集中的离散数据分别在第1, 2, 3维
            discrete_idx = [1, 2, 3]
            for idx in discrete_idx:
                one_hot, one_hot_dict = self.one_hot_encoding(data[idx])
                self.features.append(one_hot)
                self.model_map[idx] = one_hot_dict
            # 将特征串联起来
            self.features = np.concatenate(self.features, axis=1)

            # 对连续的数据进行归一化处理
            data = data.loc[:,: 40]
            continuous_idx = [i for i in range(41) if i not in discrete_idx]
            for idx in continuous_idx:
                norm_series, transform = self.max_min_normalize(data[idx])
                # 将处理后的数据添加到特征中
                self.features = np.insert(self.features, -1, values=norm_series, axis=1)
                self.model_map[idx] = transform

        else:
            # 如果是已有模型了，则使用模型的参数来进行预处理
            with open(model_path, 'rb') as model_in:
                self.model_map = pickle.load(model_in)

            # 对输出的标签进行onehot处理
            input_y = [tag.rstrip().rstrip('.') for tag in data[41]]
            input_y = [class_map[i] for i in input_y]
            # 使用one_hot编码
            self.y_one_hot = self.one_hot_encoding(input_y, self.model_map['y'])

            # 这里存在one_hot编码过于稀疏的问题，后续可能可以采用其他方法代替
            # 数据集中的离散数据分别在第1, 2, 3维
            discrete_idx = [1, 2, 3]
            for idx in discrete_idx:
                one_hot = self.one_hot_encoding(data[idx], self.model_map[idx])
                self.features.append(one_hot)
            # 将特征串联起来
            self.features = np.concatenate(self.features, axis=1)

            # 对连续的数据进行归一化处理
            continuous_idx = [i for i in range(41) if i not in discrete_idx]
            for idx in continuous_idx:
                norm_series = self.max_min_normalize(data[idx], self.model_map[idx])
                # 将处理后的数据添加到特征中
                self.features = np.insert(self.features, -1, values=norm_series, axis=1)

            self.features = np.array(self.features)

    def max_min_normalize(self, series, transform=None):
        '''
        最大、最小值的均匀归一化，将数据放缩到[0, 1]区间内,
        注意这里是以训练集为标准的,所以在测试集中可能会出现在[0,1]区间外的情况。
        '''
        if not transform:
            max_num = max(series)
            min_num = min(series)
            if max_num - min_num == 0:
                num_len = 0.001
            else:
                num_len = max_num - min_num
            norm_series = [(i - min_num) / num_len for i in series]
            return norm_series, [max_num, min_num]
        else:
            max_num = transform[0]
            min_num = transform[1]
            if max_num - min_num == 0:
                num_len = 0.001
            else:
                num_len = max_num - min_num
            norm_series = [(i - min_num) / num_len for i in series]
            return norm_series

    def one_hot_encoding(self, series, transform=None):
        '''
        将输入的序列进行one_hot编码，输出为one_hot编码后的矩阵
        若输入的是测试集，则存在transform属性，onehot编码根据transform字典进行转化。
        '''
        if not transform:
            # 统计一共有多少类别
            classes = list(set(series))
            # 生成类别对应的字典
            class_dict = {c: [1 if j == c else 0 for j in classes] for c in classes}
            return np.array([class_dict[c] for c in series], dtype='float32'), class_dict
        else:
            class_dict = transform
            return np.array([class_dict[c] if c in class_dict.keys() else [0 for _ in range(len(list(transform.values())[0]))] for c in series], dtype='float32')

    def get_class_map(self):
        '''
        获取分类的输出类型对应的四种大类的对应map。
        '''
        normal = {'noramal': ['normal']}
        probe = {'probe': ['ipsweep', 'mscan', 'nmap', 'portsweep', 'saint', 'satan']}
        dos = {'dos': ['apache2', 'back', 'land', 'mailbomb', 'neptune', 'pod', 'processtable', 'smurf', 'teardrop', 'udpstorm']}
        u2r = {'u2r': ['buffer_overflow', 'httptunnel', 'loadmodule', 'perl', 'ps', 'rootkit', 'sqlattack', 'xterm']}
        r2l = {'r2l': ['ftp_write', 'guess_passwd', 'imap', 'multihop', 'named', 'phf', 'sendmail', 'snmpgetattack', 'snmpguess', 'spy', 'warezclient', 'warezclient', 'warezmaster', 'worm', 'xlock', 'xsnoop']}
        class_map = {}
        for class_type in [normal, probe, dos, u2r, r2l]:
            for type_name, s_types in class_type.items():
                for s_type in s_types:
                    class_map[s_type] = type_name
        return class_map

    def get_features(self):
        '''
        获取处理完的特征
        '''
        return self.features

    def get_y(self):
        '''
        获取one_hot 编码后的输出y
        '''
        return self.y_one_hot

    def get_model(self):
        '''
        获取各个维度的map
        '''
        return self.model_map

    def save_model(self, model_path):
        with open(model_path, 'wb') as model_out:
            pickle.dump(self.model_map, model_out)


if __name__ == '__main__':
    pre = Preproccessor('data/kddcup.data_10_percent')
    preprocessed_path = './data/preprocessed/'
    if not os.path.exists(preprocessed_path):
        os.mkdir(preprocessed_path)

    # 保存特征
    features = pre.get_features()
    np.save(os.path.join(preprocessed_path, 'features.npy'), features)

    # 保存one hot编码后的分类
    y = pre.get_y()
    np.save(os.path.join(preprocessed_path, 'y.npy'), y)

    # 保存处理特征时的模型
    model_map = pre.get_model()
    with open(os.path.join(preprocessed_path, 'preprocess_model.pkl'), 'wb') as model_out:
        pickle.dump(model_map, model_out)
