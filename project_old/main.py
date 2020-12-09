"""
Traffic Flow Prediction with Neural Networks(SAEs、LSTM、GRU).
"""
import math
import warnings
import numpy as np
import pandas as pd
from keras.models import load_model
import sklearn.metrics as metrics
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import stats
import joblib

from preprocess_data.data import process_data

warnings.filterwarnings("ignore")


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


def MAPE_ape(y_true, y_pred):
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

    tmp=[]
    for i in range(num):
        tmp.append((abs(y[i] - y_pred[i]) / y[i])*100)

    return tmp

def plotCDF(sample, line_color, name):
    val, cnt = np.unique(sample, return_counts=True)
    pmf = cnt / len(sample)
    fs_rv_dist2 = stats.rv_discrete(name='fs_rv_dist2', values=(val, pmf))
    plt.plot(val, fs_rv_dist2.cdf(val), line_color, lw=1, alpha=0.6, label=name)




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


def plot_results(y_true, y_preds, names,linestyles):
    """Plot
    Plot the true data and predicted data.

    # Arguments
        y_true: List/ndarray, ture data.
        y_pred: List/ndarray, predicted data.
        names: List, Method names.
    """
    d = '2018/12/18 00:00'

    x = pd.date_range(d, periods=y_true.size, freq='5min')

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(x, y_true, label='True Data' , linestyle='-')
    for name, y_pred,linestyle in zip(names, y_preds ,linestyles):
        ax.plot(x, y_pred, label=name, linestyle=linestyle)

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


def main():
    lag = 64  # 16 32 64
    gap = 5  # 0 2 5
    lstm = load_model('model_old/lstm_Vehicle_lag{}.h5'.format(lag))
    a_lstm = load_model('model_old/lstm_Vehicle_attention_lag{}.h5'.format(lag))
    svr = joblib.load('model/{}lag{}gap_svr.pkl'.format(lag, gap))

    models = [lstm, a_lstm, svr]
    names = ['lstm', 'a_lstm', 'svr']
    linestyles = ['--', '-.', ':']
    file1 = 'data_old/VehicleCount_Train_workday.csv'
    file2 = 'data_old/test_mag.csv'
    _, _, X_test, y_test, scaler = process_data(file1, file2, lag, gap)
    y_test = scaler.inverse_transform(y_test.reshape(-1, 1)).reshape(1, -1)[0]
    result = []
    y_preds = []
    for name, model in zip(names, models):
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        if name == 'svr':
            X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1]))

        predicted = model.predict(X_test)
        predicted = scaler.inverse_transform(predicted.reshape(-1, 1)).reshape(1, -1)[0]
        y_preds.append(predicted[:])
        print(name)
        eva_regress(y_test, predicted)
        arr = MAPE_ape(y_test, predicted)
        result.append(np.array(arr))

    save = pd.DataFrame(np.array(result).T, columns=[names[0], names[1], names[2]])
    save.to_csv('CDF_svr{}.csv'.format(lag), index=False, header=True)
    plot_results(y_test[:], y_preds, names, linestyles)


if __name__ == '__main__':
    main()
