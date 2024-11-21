import numpy as np
from os.path import join
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


data = np.load('data/train/X_reference.npy')
label = np.load('data/train/y_reference.npy')
data = np.expand_dims(data, axis=1)
#进行插值

# 目标大小
target_size = 2048

# 原始数据的索引
original_indices = np.linspace(0, 1, data.shape[2])

# 新的插值索引
new_indices = np.linspace(0, 1, target_size)

# 初始化插值结果数组
interpolated_data = np.zeros((data.shape[0], data.shape[1], target_size))

for i in range(data.shape[0]):
    f = interp1d(original_indices, data[i, 0], kind='linear', fill_value='extrapolate')
    interpolated_data[i, 0] = f(new_indices)

print(interpolated_data.shape)
# 假设 interpolated_data 是你的插值后的数据
np.save('interpolated_data.npy', interpolated_data)

# 绘制原始数据
plt.figure(figsize=(10, 5))
plt.plot(original_indices, data[0, 0], label='Original Data', marker='o')

# 绘制插值后的数据
plt.plot(new_indices, interpolated_data[0, 0], label='Interpolated Data', linestyle='--')

plt.legend()
plt.show()