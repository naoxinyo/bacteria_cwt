import numpy as np

# 加载.npy文件
array = np.load(r'D:\CM-Net-main\data\test\y_test.npy')

# 打印文件内容
print(array)

# 查看数组的形状
print("Y_Shape of the array:", array.shape)

# 查看数组的数据类型
print("Data type of the array:", array.dtype)

# 查看数组中的元素总数
print("Total number of elements in the array:", array.size)