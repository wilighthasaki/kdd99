'''
这是对初始CSV文件进行预处理的代码，主要是对数据进行读取，对标量进行one-hot编码和对数值类型的特征进行归一化。
'''

import numpy as np
import pandas as pd
import pickle

class Preproccessor(object):
    '''
    预处理的类，初始化时数据的输入和输出路径
    '''
    def __init__(self, input_path, pca=True, model_path=None):
        # 读取数据
        data = pd.read_csv(input_path, names=[i for i in range(42)])

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
            y_one_hot, y_dict = self.one_hot_encoding(input_y)
            self.model_map['y'] = y_dict

            # 对数据集中的离散数据进行one_hot编码，同时返回字典。
            # 这里存在one_hot编码过于稀疏的问题，后续可能可以采用其他方法代替
            # 数据集中的离散数据分别在第1, 2, 3维
            discrete_idx = [1, 2, 3]
            for idx in discrete_idx:
                one_hot, one_hot_dict = self.one_hot_encoding(data[idx])
                one_hot = list(np.transpose(np.array(one_hot)))
                self.features += one_hot
                self.model_map[idx] = one_hot_dict

            # 对连续的数据进行归一化处理
#             data = data.loc[:40]
#             continuous_idx = [i for i in range(41) if i not in discrete_idx]
#             for idx in continuous_idx:
#                 norm_series, transform = self.max_min_normalize(data[idx])
#                 self.features.append(norm_series)
#                 self.model_map[idx] = transform

            self.features = np.array(self.features)
            print(self.features)
            print(self.features.shape)


    def max_min_normalize(self, series, transform=None):
        '''
        最大、最小值的归一化，将数据放缩到[0, 1]区间内,
        注意这里是以训练集为标准的,所以在测试集中可能会出现在[0,1]区间外的情况。
        '''
        if not transform:
            max_num = max(series)
            min_num = min(series)
            num_len = max_num - min_num + 0.0001
            norm_series = [(i - min_num) / num_len for i in series]
            return norm_series, [max_num, min_num]
        else:
            max_num = transform[0]
            min_num = transform[1]
            num_len = max_num - min_num + 0.0001
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
            return [class_dict[c] for c in series], class_dict
        else:
            class_dict = transform
            return [class_dict[c] for c in series]

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


if __name__ == '__main__':
    pre = Preproccessor('data/kddcup.data_10_percent')
