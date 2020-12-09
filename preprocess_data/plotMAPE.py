import matplotlib.pyplot as plt
import numpy as np
# 构建数据
x_data = ['15min&Sequence16', '15min&Sequence32','15min&Sequence64']
# y_data = [14.91, 10.44, 10.15]
# y_data2 = [12.79, 10.22, 9.74]
# y_data3 = [17.43, 13.92, 11.98]
y_data = [12.68, 12.71, 11.17]
y_data2 = [13.98, 11.92, 10.22]
y_data3 = [14.59, 13.82, 10.65]
bar_width=0.2

x = range(len(x_data))
# 将X轴数据改为使用range(len(x_data), 就是0、1、2...
plt.bar(x,y_data3 , label='SVR',
    color='#00CED1', alpha=0.8, width=bar_width)
plt.bar([i + 0.2 for i in x],y_data , label='LSTM',
    color='steelblue', alpha=0.8, width=bar_width)
# 将X轴数据改为使用np.arange(len(x_data))+bar_width,
# 就是bar_width、1+bar_width、2+bar_width...这样就和第一个柱状图并列了
plt.bar([i + 0.4 for i in x],y_data2,
    label='Attention-LSTM', color='indianred', alpha=0.8, width=bar_width)

plt.xticks([i + 0.1 for i in x], x_data)
# 为两条坐标轴设置名称
plt.xlabel("Prediction interval and Sequence length")
plt.ylabel("MAPE(%)")
# 显示图例
plt.legend()
plt.show()