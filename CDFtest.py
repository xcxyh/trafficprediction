import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

def plotCDF(sample, line_color, name):
    val, cnt = np.unique(sample, return_counts=True)
    pmf = cnt / len(sample)
    fs_rv_dist2 = stats.rv_discrete(name='fs_rv_dist2', values=(val, pmf))
    plt.plot(val, fs_rv_dist2.cdf(val), line_color, lw=1.5, alpha=0.6, label=name)



file1 = 'data_old/CDF4.csv'
mape = pd.read_csv(file1)
file2 = 'data_old/CDF5.csv'
mape2 = pd.read_csv(file2)
file3 = 'data_old/CDF6.csv'
mape3 = pd.read_csv(file3)
file_svr16 = 'CDF_svr16.csv'
mape_svr16 = pd.read_csv(file_svr16)
file_svr32 = 'CDF_svr32.csv'
mape_svr32 = pd.read_csv(file_svr32)
file_svr64 = 'CDF_svr64.csv'
mape_svr64 = pd.read_csv(file_svr64)



ALSTM_gap2_lag16 = mape['A-LSTM_gap5_lag16']
LSTM_gap2_lag16 = mape['LSTM_gap5_lag16']

ALSTM_gap2_lag32 = mape2['A-LSTM_gap5_lag32']
LSTM_gap2_lag32 = mape2['LSTM_gap5_lag32']

ALSTM_gap2_lag64 = mape3['A-LSTM_gap5_lag64']
LSTM_gap2_lag64= mape3['LSTM_gap5_lag64']

svr516 = mape_svr16['svr']
svr532 = mape_svr32['svr']
svr564 = mape_svr64['svr']


plotCDF(ALSTM_gap2_lag16,'r-', 'A-LSTM(lag16)')
plotCDF(LSTM_gap2_lag16,'r--', 'LSTM(lag16)')
plotCDF(svr516,'r-.', 'SVR(lag16)')
plotCDF(ALSTM_gap2_lag32,'g-', 'A-LSTM(lag32)')
plotCDF(LSTM_gap2_lag32,'g--', 'LSTM(lag32)')
plotCDF(svr532,'g-.', 'SVR(lag32)')
plotCDF(ALSTM_gap2_lag64,'b-', 'A-LSTM(lag64)')
plotCDF(LSTM_gap2_lag64,'b--', 'LSTM(lag64)')
plotCDF(svr564,'b-.', 'SVR(lag64)')





plt.xlabel('Mean Absolute Error (MAE)')
plt.ylabel('Empirical Cumulative Distribution Function (CDF)')

plt.axis([0,50,0,1])
plt.legend(loc='best')
plt.show()