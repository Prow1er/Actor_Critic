import numpy as np

from strategy_library import *
from search import *
from indexinfo import *


def cal_one_department(score_list: list[list[int]], idx: indexinfo):
    if 20 <= idx.outlier < 50:
        idx.max_outlier = 50
        idx.min_outlier = 20
    if 10 <= idx.outlier < 20:
        idx.max_outlier = 20
        idx.min_outlier = 10
    if 5 <= idx.outlier < 10:
        idx.max_outlier = 10
        idx.min_outlier = 5
    if 0 <= idx.outlier < 5:
        idx.max_outlier = 5
        idx.min_outlier = 0
    if -5 <= idx.outlier < 0:
        idx.max_outlier = 0
        idx.min_outlier = -5
    if -10 <= idx.outlier < -5:
        idx.max_outlier = -5
        idx.min_outlier = -10
    if -20 <= idx.outlier < -10:
        idx.max_outlier = -10
        idx.min_outlier = -20
    if -50 <= idx.outlier < -20:
        idx.max_outlier = -20
        idx.min_outlier = -50

    first_elements = []
    second_elements = []
    third_elements = []

    for i in range(len(score_list)):
        if not score_list[i]:
            score_list[i] = [0, 0, 0]
    #print('scl',score_list)
    for i in score_list:
        first_elements.append(i[0])
        second_elements.append(i[1])
        third_elements.append(i[2])

    weight = fetch_data((idx.max_outlier - idx.min_outlier),
                        [first_elements, second_elements, third_elements])
    # print(weight)
    return weight


def set_groups(list_of_strategy_group: list[Strategy_Group], idx: indexinfo):
    dept1, dept2, dept3, dept4, dept5, dept6, dept7 = [], [], [], [], [], [], []
    dept = [dept1, dept2, dept3, dept4, dept5, dept6, dept7]
    for group in list_of_strategy_group:
        dept1.append(group.strategies.get(1, []))
        dept2.append(group.strategies.get(2, []))
        dept3.append(group.strategies.get(3, []))
        dept4.append(group.strategies.get(4, []))
        dept5.append(group.strategies.get(5, []))
        dept6.append(group.strategies.get(6, []))
        dept7.append(group.strategies.get(7, []))

    # for i in range(len(dept)):
    #     print(dept[i])
    if dept1 != []:
        dept1 = cal_one_department(dept1, idx)
    else:
        dept1 = []
    if dept2 != []:
        dept2 = cal_one_department(dept2, idx)
    else:
        dept2 = []
    if dept3 != []:
        dept3 = cal_one_department(dept3, idx)
    else:
        dept3 = []
    if dept4 != []:
        dept4 = cal_one_department(dept4, idx)
    else:
        dept4 = []
    if dept5 != []:
        dept5 = cal_one_department(dept5, idx)
    else:
        dept5 = []
    if dept6 != []:
        dept6 = cal_one_department(dept6, idx)
    else:
        dept6 = []
    if dept7 != []:
        dept7 = cal_one_department(dept7, idx)
    else:
        dept7 = []
    dept = [dept1, dept2, dept3, dept4, dept5, dept6, dept7]
    # for i in range(len(dept)):
    #     print(i, dept[i])

    sum1, count = [], []
    length = len(list_of_strategy_group)
    for i in range(length):
        sum1.append(0)
        count.append(0)
    # print(sum1)
    for j in range(7):
        for i in range(length):
            sum1[i] += dept[j][i]
            if dept[j][i] != 0:
                count[i] += 1
    # print(sum1, count)
    avg = []
    for i in range(len(sum1)):
        if count[i] != 0:
            avg.append(sum1[i] / count[i])
        else:
            avg.append(0)
    # print("avg",avg)
    for i in range(len(avg)):
        list_of_strategy_group[i].effect = avg[i]
        # print(list_of_strategy_group[i].name, list_of_strategy_group[i].effect)
    for i in range(len(avg)):
        # print(idx.max_outlier, idx.min_outlier)
        if np.isnan(list_of_strategy_group[i].effect):
            list_of_strategy_group[i].effect = idx.max_outlier - idx.min_outlier
    list_of_strategy_group = sorted(list_of_strategy_group, key=lambda x: abs(x.effect), reverse=True)

    # print(list_of_strategy_group)
    return list_of_strategy_group


def get_groups_list(groups_list, index):
    for key in range(8):
        if key not in groups_list:
            groups_list.update({key: []})

    # 定义不同偏移区间的策略
    for key, value in groups_list.items():
        groups_list[key] = set_groups(value, index)
    #     print(key, groups_list[key])
    return groups_list

if __name__ == '__main__':
    idx = indexinfo(1, '利税比', -12, 2, -2)
    groups_list = fetch_strategy_groups(idx.id)
    groups_list = get_groups_list(groups_list, idx)
    for key, value in groups_list.items():
        print(key, groups_list[key])
