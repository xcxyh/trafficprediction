
import pandas
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib as mpl
import sklearn.metrics as metrics
#设置横纵坐标的名称以及对应字体格式
mpl.rcParams['font.sans-serif'] = ['SimHei']
font2 = {'family' : 'SimHei',
'weight' : 'normal',
'size'   : 12,
}
# all = pandas.read_csv('liuliang.csv',engine='python', skipfooter=3)
# all = pandas.read_csv('liuliang.csv',engine='python', skipfooter=3)
all = pandas.read_csv('occupancy.csv',engine='python', skipfooter=3)
one = all['rawOccupancy']
# two = all['error']
# three = all['lost']
four = all['ESM_0.3']
five = all['AESM']
six = all['MAM']

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(one,'-x',label='原始数值')
# ax.plot(two, '*', label='error')
# ax.plot(three, '.', label='loss')
ax.plot(four, '.', label=U'指数平滑（0.3）')
ax.plot(five, '^', label=U'自适应指数平滑')
ax.plot(six, '*', label='移动平均（分钟）')
plt.legend()
plt.grid(True)
plt.xlabel('序列')
plt.ylabel('占道时长')
# plt.savefig('xds_fig2.png')
