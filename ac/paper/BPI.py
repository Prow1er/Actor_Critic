import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 16
plt.rcParams['axes.unicode_minus'] = False


# def rer(a):
#     return (-49.79325504261936) / a
#
#
# def tir(t):
#     return (60 - t) / 60
#
#
# def bpi(a, b):
#     return 0.8 * a + 0.2 * b
#
#
# m = -1128.6170076932533
# n = 51.29958152770996
# x = rer(m)
# y = tir(n)
# print(bpi(x, y)*100)



# 数据
data = np.array([
    [56.753, 74.864, 99.442, 99.325, 99.24, 98.737],  # MAC
    [30.923, 33.175, 46.701, 60.649, 86.238, 99.064],  # AC
    [20.356, 20.338, 18.23, 15.569, 13.434, 6.43]     # DQN
])

# 定义行和列标签
algorithms = ['MAC', 'AC', 'DQN']
data_sizes = ['10', '20', '40', '60', '100', '200']

# 创建热图
plt.figure(figsize=(8, 6))
sns.heatmap(data, annot=True, fmt=".3f", cmap="YlGnBu", xticklabels=data_sizes, yticklabels=algorithms)

# 设置标题和标签
plt.title('BPI Contrast')
plt.xlabel('Training Data Volume')
plt.ylabel('Algorithm')

# 显示图表
plt.show()

