import numpy as np

data_list=[[1,2,3],[1,2,1],[3,4,5],[4,5,6],[5,6,7],[6,7,8],[6,7,9],[0,4,7],[4,6,0],[2,9,1],[5,8,7],[9,7,8],[3,7,9]]
data_list = np.array(data_list)  # 列表变数组
print(data_list[:,:-1])   # 切片出除最后一列的所有数据

print(data_list[:,-1])    # 切片出最后一列
