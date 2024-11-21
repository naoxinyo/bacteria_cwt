import torch

# 加载.pt文件
checkpoint = torch.load('D:/CM-Net-main/data/test/test_data.pt')

# 打印文件内容
print(checkpoint)

# 查看张量的形状
print(checkpoint.shape)

# 查看张量的数据类型
print(checkpoint.dtype)

# 查看张量的元素数量
num_elements = checkpoint.numel()
print("Total number of elements in the tensor:", num_elements)
