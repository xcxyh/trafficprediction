"""
Processing the data
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
def movingaverage(data, window_size):
    window = np.ones(int(window_size))/float(window_size)
    return np.convolve(data, window, 'same')

def process_data(train, test, lags, gap):
    """Process data
    Reshape and split train\test data.

    # Arguments
        train: String, name of .csv train file.
        test: String, name of .csv test file.
        lags: integer, time lag.
    # Returns
        X_train: ndarray.
        y_train: ndarray.
        X_test: ndarray.
        y_test: ndarray.
        scaler: StandardScaler.
    """
    attr = 'vehicleCount'
    df1 = pd.read_csv(train, encoding='utf-8').fillna(0)
    df2 = pd.read_csv(test, encoding='utf-8').fillna(0)
    df1[attr] = movingaverage(data=df1[attr].values, window_size=3)
    df2[attr] = movingaverage(data=df2[attr].values, window_size=3)
    # scaler = StandardScaler().fit(df1[attr].values)
    scaler = MinMaxScaler(feature_range=(0, 1)).fit(df1[attr].values.reshape(-1, 1))   # 对每一列数据归一化 到 【0，1】
    flow1 = scaler.transform(df1[attr].values.reshape(-1, 1)).reshape(1, -1)[0]
    flow2 = scaler.transform(df2[attr].values.reshape(-1, 1)).reshape(1, -1)[0]

    train, test = [], []
    for i in range(lags+gap, len(flow1)):
        train.append(flow1[i - lags-gap: i + 1])
    for i in range(lags+gap, len(flow2)):
        test.append(flow2[i - lags-gap: i + 1])

    train = np.array(train)
    test = np.array(test)
    np.random.shuffle(train)

    X_train = train[:, :-1-gap]  # 切片出除最后一列的所有数据
    y_train = train[:, -1]   # 切片出最后一列
    X_test = test[:, :-1-gap]
    y_test = test[:, -1]

    return X_train, y_train, X_test, y_test, scaler
