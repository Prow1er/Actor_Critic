import numpy as np
from DQN import *
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.font_manager as fm

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 16
plt.rcParams['axes.unicode_minus'] = False

mean_MAC = np.array([-107.05179, -71.950, -49.79, -49.79, -49.79, -49.79])
mean_AC = np.array([-361.3956, -298.556, -148.075, -97.2028, -59.7527, -49.79])
mean_DQN = np.array([-11039.29587, -11595.7233, -10223.21, -9614.9386, -4140.111, -1128.617])

time_MAC = np.array([1.3740, 1.4990, 1.6749, 2.024504, 2.279854, 3.7886424064])
time_AC = np.array([0.2992, 0.50259, 0.6020, 0.994806, 1.283509, 2.807828])
time_DQN = np.array([0.0141, 0.01800, 6.55299, 14.53663, 22.584005, 51.29958])

# 自定义x轴的标签
x_labels = ['10', '20', '40', '60', '100', '200']

# 创建图形和子图
fig, ax = plt.subplots(2, 2, figsize=(8, 5))

# 绘制第一张图 - 均值
ax[0, 0].plot(mean_MAC, label='MAC', marker='o')
ax[0, 0].plot(mean_AC, label='AC', marker='s')
ax[0, 0].set_title('MAC/AC Average Reward')
ax[0, 0].set_xlabel('Training Data Volume')
ax[0, 0].set_ylabel('Task Reward')
ax[0, 0].set_xticks(range(len(x_labels)))  # 设置x轴刻度
ax[0, 0].set_xticklabels(x_labels)  # 设置x轴标签
# ax1.set_yscale('symlog')
ax[0, 0].legend()
ax[0, 0].grid(True)

# 绘制第二张图 - 时间消耗
ax[1, 0].plot(time_MAC, label='MAC', marker='o')
ax[1, 0].plot(time_AC, label='AC', marker='s')
ax[1, 0].set_title('MAC/AC Elapsed Time')
ax[1, 0].set_xlabel('Training Data Volume')
ax[1, 0].set_ylabel('Time(seconds)')
ax[1, 0].set_xticks(range(len(x_labels)))  # 设置x轴刻度
ax[1, 0].set_xticklabels(x_labels)  # 设置x轴标签
# ax2.set_yscale('symlog')
ax[1, 0].legend()
ax[1, 0].grid(True)

ax[0, 1].plot(mean_MAC, label='MAC', marker='o')
ax[0, 1].plot(mean_DQN, label='DQN', marker='^')
ax[0, 1].set_title('MAC/DQN Average Reward')
ax[0, 1].set_xlabel('Training Data Volume')
ax[0, 1].set_ylabel('Task Reward')
ax[0, 1].set_xticks(range(len(x_labels)))  # 设置x轴刻度
ax[0, 1].set_xticklabels(x_labels)  # 设置x轴标签
# ax1.set_yscale('symlog')
ax[0, 1].legend()
ax[0, 1].grid(True)

ax[1, 1].plot(time_MAC, label='MAC', marker='o')
ax[1, 1].plot(time_DQN, label='DQN', marker='^')
ax[1, 1].set_title('MAC/DQN Elapsed Time')
ax[1, 1].set_xlabel('Training Data Volume')
ax[1, 1].set_ylabel('Time(seconds)')
ax[1, 1].set_xticks(range(len(x_labels)))  # 设置x轴刻度
ax[1, 1].set_xticklabels(x_labels)  # 设置x轴标签
# ax2.set_yscale('symlog')
ax[1, 1].legend()
ax[1, 1].grid(True)

# 显示图形
plt.tight_layout()
plt.show()
