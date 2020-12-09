"""
Train the NN model.
"""
import sys
import warnings
import argparse
import numpy as np
import pandas as pd
from keras.models import Model
import joblib

import model

warnings.filterwarnings("ignore")


def train_model(model, X_train, y_train, name, config):
    """train
    train a single model.

    # Arguments
        model: Model, NN model to train.
        X_train: ndarray(number, lags), Input data for train.
        y_train: ndarray(number, ), result data for train.
        name: String, name of model.
        config: Dict, parameter for train.
    """

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


def train_seas(models, X_train, y_train, name, config):
    """train
    train the SAEs model.

    # Arguments
        models: List, list of SAE model.
        X_train: ndarray(number, lags), Input data for train.
        y_train: ndarray(number, ), result data for train.
        name: String, name of model.
        config: Dict, parameter for train.
    """

    temp = X_train
    # early = EarlyStopping(monitor='val_loss', patience=30, verbose=0, mode='auto')

    for i in range(len(models) - 1):
        if i > 0:
            p = models[i - 1]
            hidden_layer_model = Model(input=p.input,
                                       output=p.get_layer('hidden').output)
            temp = hidden_layer_model.predict(temp)

        m = models[i]
        # 损失函数为 mse 均方误差  优化器 rmsprop  指标列表metrics：平均绝对百分比误差 ，缩写MAPE。
        m.compile(loss="mse", optimizer="rmsprop", metrics=['mape'])

        m.fit(temp, y_train, batch_size=config["batch"],
              epochs=config["epochs"],
              validation_split=0.05)

        models[i] = m

    saes = models[-1]
    for i in range(len(models) - 1):
        weights = models[i].get_layer('hidden').get_weights()
        saes.get_layer('hidden%d' % (i + 1)).set_weights(weights)

    train_model(saes, X_train, y_train, name, config)


def main(argv):
    lag = 64  # 16 32 64
    gap = 5  # 2 5
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="svr_gap{}_lag_{}".format(gap, lag),
        help="Model to train.")
    args = parser.parse_args()
    config = {"batch": 64, "epochs": 10}
    file1 = 'data_old/VehicleCount_Train_workday.csv'
    file2 = 'data_old/VehicleCount_Test_workday.csv'
    X_train, y_train, _, _, _ = process_data(file1, file2, lag, gap)

    if args.model == 'lstm_gap{}_lag_{}'.format(gap, lag):
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        m = model.get_lstm([lag, 64, 64, 1])
        train_model(m, X_train, y_train, args.model, config)
    if args.model == 'attentionlstm_gap{}_lag_{}'.format(gap, lag):
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        m = model.get_attention_lstm([lag, 64, 64, 1])
        train_model(m, X_train, y_train, args.model, config)
    if args.model == 'gru_gap{}_lag_{}'.format(gap, lag):
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        m = model.get_gru([lag, 64, 64, 1])
        train_model(m, X_train, y_train, args.model, config)
    if args.model == 'saes_gap{}_lag_{}'.format(gap, lag):
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1]))
        m = model.get_saes([lag, 400, 400, 400, 1])
        train_seas(m, X_train, y_train, args.model, config)
    if args.model == 'svr_gap{}_lag_{}'.format(gap, lag):
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1]))
        m = model.get_svr(2, 0.1)
        m.fit(X_train, y_train)
        # 保存成sklearn自带的文件格式
        joblib.dump(m, 'model/' + str(lag) + 'lag' + str(gap) + 'gap_' + 'svr.pkl')

if __name__ == '__main__':
     main(sys.argv)
