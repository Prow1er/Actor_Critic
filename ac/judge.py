import numpy as np
from scipy.linalg import eig


def improved_ahp(judgment_matrix):
    # 特征向量法
    eigenvalues, eigenvectors = eig(judgment_matrix)
    max_eigenvalue_index = np.argmax(eigenvalues)
    feature_vector_weights = eigenvectors[:, max_eigenvalue_index]
    feature_vector_weights /= np.sum(feature_vector_weights)
    # print("特征：")
    # print(feature_vector_weights)

    # 归一化判断矩阵
    normalized_matrix = judgment_matrix / np.sum(judgment_matrix, axis=0)
    row_sum = np.sum(normalized_matrix, axis=1)
    # 算术平均法
    arithmetic_mean_weights = row_sum / judgment_matrix.shape[1]
    # print("算术：")
    # print(arithmetic_mean_weights)

    # 几何平均法
    geometric_mean_weights = np.power(np.prod(judgment_matrix, axis=1), 1 / judgment_matrix.shape[0])
    geometric_mean_weights /= np.sum(geometric_mean_weights)
    # print("几何：")
    # print(geometric_mean_weights)

    # 平均权重
    average_weights = (arithmetic_mean_weights + geometric_mean_weights + feature_vector_weights) / 3
    # print("平均权重：")
    # print(average_weights)

    return average_weights


def critic_method(data):
    # 计算数据在每个特征上的平均值
    mean_data = np.mean(data, axis=0)

    # 计算数据在每个特征上的方差
    variance = np.var(data, axis=0)

    # 计算每个特征的变异系数，即方差与平均值的比值
    contrast = variance / mean_data

    # 计算数据的相关矩阵
    # print(data)
    # print(data.T)
    correlation_matrix = np.corrcoef(data.T)
    # print("juzhen:")
    # print(correlation_matrix)
    # print()
    # 将相关矩阵中值为1的元素（完全相关的元素）设置为0
    # 这是因为完全相关的特征在critic方法中不提供额外信息

    if type(correlation_matrix) is not np.ndarray:
        correlation_matrix = np.array(correlation_matrix)
    correlation_matrix[correlation_matrix == 1] = 0

    # 计算每个特征的冲突度，即所有相关系数的绝对值之和
    conflict = np.sum(np.abs(correlation_matrix), axis=0)

    # 计算每个特征的信息承载量，即变异系数与冲突度的乘积
    critic_weights = contrast * conflict

    # 归一化权重，确保它们的和为1
    critic_weights /= np.sum(critic_weights)

    return critic_weights


def comparison_matrix(scores):
    n = len(scores)
    # 构建比较矩阵
    comparison_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if scores[i] != scores[j]:
                if scores[i] == 0:
                    comparison_matrix[i, j] = 0.01
                elif scores[j] == 0:
                    comparison_matrix[i, j] = scores[i] / 0.01
                else:
                    comparison_matrix[i, j] = scores[i] / scores[j]
            else:
                if scores[i] == 0 or scores[j] == 0:
                    comparison_matrix[i, j] = 0.01
                else:
                    comparison_matrix[i, j] = 1

    # while comparison_matrix.shape != (3, 3):
    #     comparison_matrix = np.pad(comparison_matrix, pad_width=1, mode='constant', constant_values=0)
    #     continue

    # print(comparison_matrix)
    return comparison_matrix


def check_zeros(matrix):
    for row in matrix:
        if 0 in row:
            return False
    return True


class Judge:
    def __init__(self, interval, scores_criterion_1, scores_criterion_2, scores_criterion_3):
        """
        初始化函数。
        :param interval: 异常区间大小
        :param scores_criterion_1: 准则一的评分
        :param scores_criterion_2: 准则二的评分
        :param scores_criterion_3: 准则三的评分
        """
        self.scores_criterion_3 = scores_criterion_3
        self.scores_criterion_2 = scores_criterion_2
        self.scores_criterion_1 = scores_criterion_1
        self.interval = interval

    def final_weights(self):
        len0 = len(self.scores_criterion_1)
        if (check_zeros(comparison_matrix(self.scores_criterion_1)) and
                check_zeros(comparison_matrix(self.scores_criterion_2)) and
                check_zeros(comparison_matrix(self.scores_criterion_3))):
            weights_criterion_1 = improved_ahp(comparison_matrix(self.scores_criterion_1))
            weights_criterion_2 = improved_ahp(comparison_matrix(self.scores_criterion_2))
            weights_criterion_3 = improved_ahp(comparison_matrix(self.scores_criterion_3))
        else:
            weights_criterion_1 = np.zeros(len0)
            weights_criterion_2 = np.zeros(len0)
            weights_criterion_3 = np.zeros(len0)
        # 将每个准则的权重相加
        ahp_weights = (weights_criterion_1 + weights_criterion_2 + weights_criterion_3) / 3
        # print(np.array(self.scores_criterion_1).reshape(-1, 1))
        # print("ahp",ahp_weights,type(ahp_weights))
        # critic
        if (check_zeros(comparison_matrix(self.scores_criterion_1)) and
                check_zeros(comparison_matrix(self.scores_criterion_2)) and
                check_zeros(comparison_matrix(self.scores_criterion_3))):
            critic_matrix = np.vstack((np.array(self.scores_criterion_1),
                                       np.array(self.scores_criterion_2),
                                       np.array(self.scores_criterion_3)))

            # critic_matrix = np.where(np.isnan(critic_matrix), 0, critic_matrix)

            critic_weights = critic_method(critic_matrix)
            critic_weights = np.where(np.isnan(critic_weights), 0, critic_weights)
        else:
            critic_weights = np.zeros(len0)
        if len(critic_weights) < len0:
            critic_weights = np.zeros(len0)

        # print("critic",critic_weights,type(critic_weights))
        total_weights = ahp_weights * 0.8 + critic_weights * 0.2
        # total_weights = np.where(np.isnan(total_weights), 0, total_weights)
        # 归一化总权重
        if np.sum(total_weights) != 0:
            final_weights = total_weights / np.sum(total_weights)
        else:
            final_weights = np.zeros(len0)
        final_weights = np.where(np.isnan(final_weights), 0, final_weights)
        # print(final_weights)
        # 计算得分
        # score_strategy = final_weights
        score_strategy = self.interval * final_weights

        return score_strategy


if __name__ == '__main__':
    judge = Judge(10, [5,5,4], [5,4,8], [4,9,5])
    score_strategy = judge.final_weights()
    print("score_strategy:", score_strategy)

    # [3.451007436455655, 3.312791614171911, 3.236200949372433]
    # 部门A score_strategy: [4.11209142 3.12921093 2.75869765]
    # 部门B score_strategy: [3.11181726 3.32219726 3.56598548]
    # 部门C score_strategy: [3.12911363 3.48696666 3.38391971]

"""
a a11 a12..
b a21       
c a31
d

给出的策略来验证

 策略打分表 -> 策略计算
"""
