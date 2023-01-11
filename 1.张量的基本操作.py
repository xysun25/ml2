

import numpy as np
import torch
import torchvision
print(torch.__version__)
print('gpu:', torch.cuda.is_available())

data = [[1, 2],[3, 4]]    # 嵌套列表
x_data = torch.tensor(data)    # 列表转化为张量

# 数组转化为张量，张量数据便于进行做后期深度学习数据模型输入
np_array = np.array(data)
x_np = torch.from_numpy(np_array)

# 创建张量
x_ones = torch.ones_like(x_data) # retains the properties of x_data
print(f"Ones Tensor: \n {x_ones} \n")

x_rand = torch.rand_like(x_data, dtype=torch.float) # overrides the datatype of x_data
print(f"Random Tensor: \n {x_rand} \n")


# 通过形状构建张量
shape = (2,3,)

rand_tensor = torch.rand(shape)   # 通过形状创建张量
ones_tensor = torch.ones(shape)   # 创建单位张量，都是1的数组
zeros_tensor = torch.zeros(shape)   # 创建初始化张量

# 创建随机张量
print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor}")

tensor = torch.rand(3,4)

print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")


# We move our tensor to the GPU if available  判断cuda是否可用，条件判断
if torch.cuda.is_available():
    tensor = tensor.to("cuda")

# 创建张量
tensor = torch.ones(4, 4)
print(f"First row: {tensor[0]}")
print(f"First column: {tensor[:, 0]}")
print(f"Last column: {tensor[..., -1]}")
tensor[:,1] = 0
print(tensor)

t1 = torch.cat([tensor, tensor, tensor], dim=0)   # 三个张量纵向堆叠
# df = pd.concat([], axis=1)
print(t1)

# 张量的计算
# This computes the matrix multiplication between two tensors.
# y1, y2, y3 will have the same value
y1 = tensor @ tensor.T   # @：矩阵相乘
y2 = tensor.matmul(tensor.T)   # 张量转置
y3 = torch.rand_like(tensor)
torch.matmul(tensor, tensor.T, out=y3)   # matmul和@等价，是matrix multiplication缩写：矩阵相乘

# This computes the element-wise product.
# z1, z2, z3 will have the same value
z1 = tensor * tensor
z2 = tensor.mul(tensor)
z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3)

# 张量加和
agg = tensor.sum()
agg_item = agg.item()
print(agg_item, type(agg_item))

# 张量所有值都加一
print(f"{tensor} \n")
tensor.add_(5)
print(tensor)

t = torch.ones(5)
print(f"t: {t}")

n = t.numpy()
print(f"n: {n}")

t.add_(1)
print(f"t: {t}")
print(f"n: {n}")


n = np.ones(5)
t = torch.from_numpy(n)
np.add(n, 1, out=n)


print(f"t: {t}")
print(f"n: {n}")




