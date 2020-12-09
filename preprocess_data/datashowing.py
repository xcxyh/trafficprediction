import pandas
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib as mpl
import sklearn.metrics as metrics
# 等同于MATLAB中的smooth函数，但是平滑窗口必须为奇数。

def plot_results(y_true, y_preds, names):
    """Plot
    Plot the true data and predicted data.

    # Arguments
        y_true: List/ndarray, ture data.
        y_pred: List/ndarray, predicted data.
        names: List, Method names.
    """
    d = '2018/12/18 0:00'
    x = pd.date_range(d, periods=1528, freq='5min')

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(x, y_true, label='True Data')
    for name, y_pred in zip(names, y_preds):
        ax.plot(x, y_pred, label=name)

    plt.legend()
    plt.grid(True)
    plt.xlabel('Time of Day')
    plt.ylabel('Flow')

    date_format = mpl.dates.DateFormatter("%H:%M")
    ax.xaxis.set_major_formatter(date_format)
    fig.autofmt_xdate()

    plt.show()

def MAPE(y_true, y_pred):
    """Mean Absolute Percentage Error
    Calculate the mape.

    # Arguments
        y_true: List/ndarray, ture data.
        y_pred: List/ndarray, predicted data.
    # Returns
        mape: Double, result data for train.
    """

    y = [x for x in y_true if x > 0]
    y_pred = [y_pred[i] for i in range(len(y_true)) if y_true[i] > 0]

    num = len(y_pred)
    sums = 0

    for i in range(num):
        tmp = abs(y[i] - y_pred[i]) / y[i]
        sums += tmp

    mape = sums * (100 / num)

    return mape


def eva_regress(y_true, y_pred):
    """Evaluation
    evaluate the predicted resul.

    # Arguments
        y_true: List/ndarray, ture data.
        y_pred: List/ndarray, predicted data.
    """

    mape = MAPE(y_true, y_pred)
    vs = metrics.explained_variance_score(y_true, y_pred)
    mae = metrics.mean_absolute_error(y_true, y_pred)
    mse = metrics.mean_squared_error(y_true, y_pred)
    r2 = metrics.r2_score(y_true, y_pred)
    print('explained_variance_score:%f' % vs)
    print('mape:%f%%' % mape)
    print('mae:%f' % mae)
    print('mse:%f' % mse)
    print('rmse:%f' % math.sqrt(mse))
    print('r2:%f' % r2)


def smooth(a,WSZ):
    # a:原始数据，NumPy 1-D array containing the data to be smoothed
    # 必须是1-D的，如果不是，请使用 np.ravel()或者np.squeeze()转化
    # WSZ: smoothing window size needs, which must be odd number,
    # as in the original MATLAB implementation
    out0 = np.convolve(a,np.ones(WSZ,dtype=int),'valid')/WSZ
    r = np.arange(1,WSZ-1,2)
    start = np.cumsum(a[:WSZ-1])[::2]/r
    stop = (np.cumsum(a[:-WSZ:-1])[::2]/r)[::-1]
    return np.concatenate((  start , out0, stop  ))

# another one，边缘处理的不好


def movingaverage(data, window_size):
    window = np.ones(int(window_size))/float(window_size)
    return np.convolve(data, window, 'same')


# another one，速度更快
# 输出结果 不与原始数据等长，假设原数据为m，平滑步长为t，则输出数据为m-t+1

# def movingaverage(data, window_size):
#     cumsum_vec = np.cumsum(np.insert(data, 0, 0))
#     ma_vec = (cumsum_vec[window_size:] - cumsum_vec[:-window_size]) / window_size
#     return ma_vec

def plot_loss():
    loss = pandas.read_csv('lstm_mag loss.csv', usecols=[0, 2], engine='python', skipfooter=3)
    losses = loss['loss']
    val_loss = loss['val_loss']
    names = ['galwlr']
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(losses, label='loss')
    ax.plot(val_loss, label='val_loss')
    plt.legend()
    plt.grid(True)
    plt.xlabel('epoch')
    plt.ylabel('loss')


dataset = pandas.read_csv('train_mag.csv', usecols=[2, 3], engine='python', skipfooter=3)
print(dataset['vehicleCount'])
y_true=dataset['vehicleCount']
y_preds = []
y_preds.append(dataset['vehicleCount_pre'])
eva_regress(y_true, dataset['vehicleCount_pre'])
names = ['galwlr']
plot_results(y_true, y_preds, names)

plt.subplot(211)
plt.plot(dataset)
plt.subplot(212)
mov_data =movingaverage(data=dataset['vehicleCount'].values, window_size=5)
plt.plot(mov_data)
plt.show()

