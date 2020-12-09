# -*- coding: utf-8 -*-
"""
Created on 2020/12/9 14:11
@author: Cong Xiong
"""
import argparse
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import model
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from error_evaluate import print_error_eval


def plot_result(y_true, y_pred, name):
    d = '2018/12/18 00:00'
    x = pd.date_range(d, periods=y_true.size, freq='5min')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x, y_true, label='True Data', linestyle='-')
    ax.plot(x, y_pred, label=name, linestyle='-')
    plt.legend(loc='upper left')
    # plt.grid(True)
    plt.xlabel('Time of Day')
    plt.ylabel('Traffic Flow(veh/5 min)')
    date_format = mpl.dates.DateFormatter("%H:%M")
    ax.xaxis.set_major_formatter(date_format)
    fig.autofmt_xdate()
    plt.show()


def movingaverage(data, window_size):
    window = np.ones(int(window_size))/float(window_size)
    return np.convolve(data, window, 'same')


def generate_train_test(data_set, len_test, input_len, lag):
    data_set = movingaverage(data=data_set, window_size=3)
    scaler = MinMaxScaler(feature_range=(0, 1)).fit(data_set.reshape(-1, 1))   # 对每一列数据归一化 到 【0，1】
    flow = scaler.transform(data_set.reshape(-1, 1)).reshape(1, -1)[0]
    train, test = [], []
    for i in range(input_len + lag, len(flow)):
        train.append(flow[i - input_len - lag: i + 1])
    train = np.array(train)
    # np.random.shuffle(train)
    # print(train.shape)
    x_train = train[:, :-1 - lag]  # 切片出除最后一列的所有数据
    y_train = train[:, -1]   # 切片出最后一列
    # 增加 1 维
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    # 划分 train  和 test
    X_train, X_test = x_train[:-len_test], x_train[-len_test:]
    Y_train, Y_test = y_train[:-len_test], y_train[-len_test:]
    print('X_train shape: {}'.format(X_train.shape))
    print('Y_train shape: {}'.format(Y_train.shape))
    print('X_test shape: {}'.format(X_test.shape))
    print('Y_test shape: {}'.format(Y_test.shape))
    return X_train, Y_train, X_test, Y_test, scaler


def train_model(model, X_train, y_train, name, config):
    model.compile(loss="mse", optimizer="adadelta", metrics=['mape'])
    # early = EarlyStopping(monitor='val_loss', patience=30, verbose=0, mode='auto')
    hist = model.fit(
        X_train, y_train,
        batch_size=config["batch"],
        epochs=config["epochs"],
        validation_split=0.1)  # 训练中

    model.save('models/' + name + '.h5')
    # df = pd.DataFrame.from_dict(hist.history)
    # df.to_csv('models/' + name + ' loss.csv', encoding='utf-8', index=False)
    return model


def train_and_save():
    # parameter set
    # 输入数据长度
    input_len = 16  # 16 32 64
    # 预测 滞后 时间 5 分钟间隔  2 代表 预测 15 分钟后的值
    lag = 0  # 0 2 5
    model_name = 'LSTM'   # Attention-LSTM
    # 一天的长度为 288
    len_test = 288
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="{}_input_len{}_lag{}".format(model_name, input_len, lag))
    args = parser.parse_args()
    config = {"batch": 16, "epochs": 5}
    print('==========parameter setting done==========')
    # load data set
    file_name = 'data/14806.csv'
    df1 = pd.read_csv(file_name, encoding='utf-8').fillna(0)
    data_set = df1['vehicleCount'].values
    print('==========load data set done==========')

    X_train, y_train, X_test, y_test, scaler = generate_train_test(data_set, len_test, input_len, lag)

    untrain_model = None
    if args.model == 'LSTM_input_len{}_lag{}'.format(input_len, lag):
        untrain_model = model.get_lstm([input_len, 64, 64, 1])
    if args.model == 'Attention-LSTM_input_len{}_lag_{}'.format(input_len, lag):
        untrain_model = model.get_attention_lstm([input_len, 64, 64, 1])

    trained_model = train_model(untrain_model, X_train, y_train, args.model, config)

    # predict and eval the error and plot the result
    y_pred = trained_model.predict(X_test)
    # 归一化后的结果要还原成流量
    y_test = scaler.inverse_transform(y_test.reshape(-1, 1)).reshape(1, -1)[0]
    y_pred = scaler.inverse_transform(y_pred.reshape(-1, 1)).reshape(1, -1)[0]

    print_error_eval(y_test, y_pred)
    plot_result(y_test, y_pred, model_name)


def main():
    train_and_save()


if __name__ == '__main__':
    main()
