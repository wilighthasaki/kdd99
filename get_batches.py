import numpy as np


def get_batches(data, batch_size, shuffle=False):
    """
    获得批量化的数据
    :param data: 保存要处理的data的列表
    :param batch_size: batch的大小
    :param shuffle: 是否打乱顺序
    :return: 一个batch的列表
    """
    batch_data_list = []

    # 决定是否打乱
    if shuffle:
        index = np.arange(len(data[0]))
        np.random.shuffle(index)
        new_data = []
        for i in data:
            new_data.append(i[index])
    else:
        new_data = data

    maxlen = len(data[0])

    for i in new_data:
        batch_data = []
        start_index = 0
        while start_index < maxlen:
            if start_index + batch_size > maxlen:
                start_index = maxlen - batch_size
            batch_data.append(i[start_index: start_index + batch_size])
            start_index += batch_size
        batch_data_list.append(batch_data)

    return batch_data_list


if __name__ == '__main__':
    X_data = np.arange(0,100)
    X_data = X_data.reshape([10, 10])
    y_data = np.array([1 if i > 0.5 else 0 for i in np.random.rand(10)])
    print('显示测试数据')
    print(X_data)
    print(y_data)

    print('测试分batch的效果')
    X_batch_list, y_batch_list = get_batches([X_data, y_data], batch_size=2)
    print(X_batch_list)
    print(y_batch_list)

    print('测试打乱数据集的batch效果')
    X_batch_list, y_batch_list = get_batches([X_data, y_data], batch_size=2, shuffle=True)
    print(X_batch_list)
    print(y_batch_list)
