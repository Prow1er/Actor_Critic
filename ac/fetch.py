import pymysql
import warnings
import numpy as np
import pandas as pd
from judge import Judge


def fetch_data(interval, criterion):
    group_weights = []
    judge = Judge(interval, criterion[0], criterion[1], criterion[2])
    department = judge.final_weights()
    # print(department)
    # group_weights.append(list(department))
    #
    # group_weights = [[x if not np.isnan(x) else 0 for x in row] for row in group_weights]
    # # print(group_weights)
    # # 计算每个位置上的元素之和
    # sums = [sum(column) for column in zip(*group_weights)]
    # non_zero_counts = [sum(1 for x in col if x != 0) for col in zip(*group_weights)]
    # # 计算平均值
    # average_group_weights = [s / count if count > 0 else 0 for s, count in zip(sums, non_zero_counts)]

    return department


# 调用函数
if __name__ == '__main__':
    avg = fetch_data(10, [[2, 7, 6], [4, 2, 2], [4, 1, 2]])


