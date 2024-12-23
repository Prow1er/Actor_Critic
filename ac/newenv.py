import gym
from gym import spaces
import numpy as np

import strategy_library
# from s_l import StrategyLibrary
from strategy_library import *
from indexinfo import *


class AnomalyMetricEnv(gym.Env):
    def __init__(self, idx: indexinfo = None, groups_list: dict[int:list[Strategy_Group]] = None,
                 normal_value=0, max_deviation=20, stability=0.2):
        super(AnomalyMetricEnv, self).__init__()

        self.normal_value = normal_value
        self.max_deviation = max_deviation
        self.state = None
        self.selected_strategies = []
        self.stability = stability
        self.idx = idx
        self.groups_list = groups_list

        # 初始化策略库
        self.strategy_library = StrategyLibrary(self.idx, self.groups_list)
        # a = self.strategy_library.get_all()
        # for key, value in a.items():
        #     print(f'key:{key},value:{value}')

        # 定义状态空间和动作空间
        self.observation_space = spaces.Box(low=-max_deviation, high=max_deviation, shape=(1,), dtype=np.float32)
        self.action_space = spaces.Discrete(len(self._get_all_strategies()))

    def _get_all_strategies(self):
        # 获取所有策略的列表
        all_strategies = self.strategy_library.get_all()
        return all_strategies

    def reset(self, val):
        # 初始化状态，随机生成一个偏离值
        self.state = val
        self.selected_strategies = []  # 重置所选策略列表
        return np.array([self.state], dtype=np.float32)

    def step(self, action):
        # 离散状况 动作空间 稳定区间
        # 弱模型  多参与者 动态
        self.strategy_library = StrategyLibrary(self.idx, self.groups_list)
        # a = self.strategy_library.get_all()
        # for key, value in a.items():
        #     print(f'key:{key},value:{value}')
        all_strategies = self._get_all_strategies()
        max_effect = []
        second_effect = []

        for key in all_strategies:
            # # print(all_strategies[key])
            # sorted_strategies = sorted(all_strategies[key], key=lambda s: abs(s.effect), reverse=True)
            # # print(sorted_strategies)
            # all_strategies[key] = sorted_strategies

            strategies = all_strategies.get(key, [])
            max_effect_strategy = strategies[0] if len(strategies) > 0 else None
            second_effect_strategy = strategies[1] if len(strategies) > 1 else None

            max_effect.append(max_effect_strategy)
            second_effect.append(second_effect_strategy)

        # 执行策略（动作），并更新状态
        assert self.action_space.contains(action), f"{action} ({type(action)}) invalid"

        # 获取当前状态对应的策略列表
        current_strategies = self.strategy_library.get_strategies(self.state)
        if not current_strategies:
            raise ValueError(f"No strategies available for state {self.state}")

        # 应用选定的策略
        selected_strategy = current_strategies[action % len(current_strategies)]

        self.state = selected_strategy.apply(value=self.state)
        self.idx.outlier = self.state

        self.selected_strategies.append(selected_strategy.name)  # 记录所选策略

        first_level, second_level = [], []

        for key, group in self.groups_list.items():
            if len(group) > 0 and hasattr(group[0], 'name') and group[0].name:
                first_level.append(group[0].name)

            if len(group) > 1 and hasattr(group[1], 'name') and group[1].name:
                second_level.append(group[1].name)

        if selected_strategy.name in first_level:
            reward = - 1 * ((self.state - self.normal_value) ** 2)  # 优化策略奖励更高
        elif selected_strategy.name in second_level:
            reward = - 50 * ((self.state - self.normal_value) ** 2)
        else:
            reward = - 100 * ((self.state - self.normal_value) ** 2)  # 次优策略奖励较低

        # 检查是否达到终止条件
        done = abs(self.state - self.normal_value) < self.stability

        # 返回新的状态、奖励、是否终止和额外信息
        return np.array([self.state], dtype=np.float32), reward, done, {}

    def render(self, mode='human'):
        # 可选的渲染方法
        print(f"Current state: {self.state}, Strategies: {self.selected_strategies}")

    def close(self):
        pass


if __name__ == "__main__":
    env = AnomalyMetricEnv()
    state = env.reset(25)
    # print(f"Initial state: {state}")

    for _ in range(300):
        action = env.action_space.sample()  # 随机选择一个动作
        state, reward, done, _ = env.step(action)
        # print(f"Reward: {reward} ")
        env.render()
        if done:
            # print("Episode finished")
            break
