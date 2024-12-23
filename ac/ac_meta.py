import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import numpy as np
from newenv import AnomalyMetricEnv
from ac_net import Actor, Critic


# 定义元学习Actor-Critic代理类
class MetaActorCriticAgent:

    # 初始化方法
    def __init__(self, env, lr=0.01, scheduler_step_size=200, scheduler_gamma=0.5):
        """
        初始化环境、网络结构、优化器及学习率调度器。

        参数:
        - env: 异常检测环境实例
        - lr: 学习率，默认0.02
        - scheduler_step_size: 学习率衰减步长，默认200
        - scheduler_gamma: 学习率衰减因子，默认0.5
        """
        self.env = env  # 环境实例
        self.state_dim = env.observation_space.shape[0]  # 状态空间维度
        self.action_dim = env.action_space.n  # 动作空间大小
        self.actor = Actor(self.state_dim, self.action_dim)# Actor网络
        self.critic = Critic(self.state_dim)  # Critic网络
        self.lr = lr  # 学习率

        # 初始化优化器和学习率调度器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.lr)
        self.actor_scheduler = StepLR(self.actor_optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma)
        self.critic_scheduler = StepLR(self.critic_optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma)

    # MAML训练方法
    def maml_train(self, val, inner_lr=0.01, gamma=0.98, batch=5):
        """
        使用MAML方法训练模型。

        参数:
        - val: 训练任务列表
        - inner_lr: 内层更新的学习率，默认0.001
        - gamma: 折扣因子，默认0.98
        - batch: 每批处理的任务数量，默认5
        """
        num_episodes = len(val)  # 任务数量
        outer_optimizer = optim.Adam(list(self.actor.parameters()) + list(self.critic.parameters()),
                                     lr=self.lr)  # 外层优化器

        episode = 0  # 当前任务索引
        while episode != num_episodes:
            # 内层更新后的参数和损失
            actor_params_list = []
            critic_params_list = []
            losses = []
            num_tasks = val[episode: (episode + batch)]  # 当前批次任务

            # 遍历批次内的每个任务
            for task in num_tasks:
                # 复制当前模型的参数
                actor_params = {name: param.clone() for name, param in self.actor.named_parameters()}
                critic_params = {name: param.clone() for name, param in self.critic.named_parameters()}

                state = self.env.reset(task)  # 重置环境并获取初始状态
                episode_reward = 0  # 初始化累积奖励

                # 执行任务直到完成
                while True:
                    action = self.select_action(state)  # 选择动作
                    next_state, reward, done, _ = self.env.step(action)  # 执行动作并获取结果

                    # 将状态转换为张量
                    state_tensor = torch.tensor(state, dtype=torch.float32)
                    next_state_tensor = torch.tensor(next_state, dtype=torch.float32)
                    reward_tensor = torch.tensor(reward, dtype=torch.float32)

                    # 计算价值和目标价值
                    value = self.critic(state_tensor)
                    next_value = self.critic(next_state_tensor)
                    target = reward_tensor + (1 - done) * gamma * next_value
                    target = target.detach()

                    # 计算损失
                    critic_loss = (target - value).pow(2).mean()
                    log_prob = torch.log(self.actor(state_tensor)[action])
                    advantage = target - value
                    actor_loss = -log_prob * advantage.detach()

                    # 内层更新（SGD或Adam）
                    actor_grads = torch.autograd.grad(actor_loss, self.actor.parameters(), retain_graph=True)
                    critic_grads = torch.autograd.grad(critic_loss, self.critic.parameters(), retain_graph=True)

                    # 使用内层学习率更新参数
                    with torch.no_grad():
                        for (name, param), grad in zip(self.actor.named_parameters(), actor_grads):
                            actor_params[name] -= inner_lr * grad
                        for (name, param), grad in zip(self.critic.named_parameters(), critic_grads):
                            critic_params[name] -= inner_lr * grad

                    # 如果任务完成，则退出循环
                    if done:
                        break

                    state = next_state  # 更新当前状态

                # 记录内层更新后的参数和损失
                actor_params_list.append(actor_params)
                critic_params_list.append(critic_params)
                losses.append(actor_loss + critic_loss)

            # 外层更新
            outer_optimizer.zero_grad()
            meta_loss = torch.stack(losses).mean()
            meta_loss.backward()
            outer_optimizer.step()

            # 更新学习率调度器
            self.actor_scheduler.step()
            self.critic_scheduler.step()
            episode += batch  # 更新任务索引

    # 测试方法
    def test(self, val):
        """
        测试模型在一组任务上的性能。

        参数:
        - val: 测试任务列表
        """
        num_episodes = len(val)  # 任务数量
        self.actor.eval()  # 设置Actor网络为评估模式
        self.critic.eval()  # 设置Critic网络为评估模式

        initial_states = []  # 初始状态列表
        final_states = []  # 最终状态列表
        strategy_sequences = []  # 策略序列列表
        rewards = []  # 奖励列表
        state_sequences = []  # 每一步的状态值列表

        # 对每个任务进行测试
        for episode in range(num_episodes):
            state = self.env.reset(val[episode])  # 重置环境并获取初始状态
            initial_states.append(state)  # 记录初始状态
            episode_reward = 0  # 初始化累积奖励
            episode_strategies = []  # 单个任务的策略序列
            episode_state_sequence = [state.item()]  # 记录当前任务的每一步状态值

            # 执行任务直到完成
            while True:
                action = self.select_action(state)  # 选择动作
                next_state, reward, done, _ = self.env.step(action)  # 执行动作并获取结果
                episode_reward += reward  # 累加奖励

                episode_strategies.append(self.env.selected_strategies[-1])  # 记录当前策略
                episode_state_sequence.append(next_state.item())  # 记录下一步状态

                if done:  # 如果任务完成
                    final_states.append(next_state)  # 记录最终状态
                    break

                state = next_state  # 更新当前状态

            # 记录测试结果
            rewards.append(episode_reward)
            strategy_sequences.append(episode_strategies)
            state_sequences.append(episode_state_sequence)  # 添加每一步的状态值列表
            # print(f"Episode {episode + 1}, "
            #       f"初始值: {initial_states[-1]}, "
            #       f"最终值: {final_states[-1]}, "
            #       f"策略: {episode_strategies}, "
            #       f"总奖励: {episode_reward:.2f}, "
            #       f"状态序列: {episode_state_sequence}")
            episode_state_sequence = [round(num, 3) for num in episode_state_sequence]
            out_str = (f"Episode {episode + 1}, "
                       f"初始值: {initial_states[-1]}, "
                       f"最终值: {final_states[-1]}, "
                       f"策略: {episode_strategies}, "
                       f"总奖励: {episode_reward:.3f}, "
                       f"状态序列: {episode_state_sequence}")
            out_dict = {
                # "Episode": episode + 1,
                "Initial value": initial_states[-1],
                "Final value": final_states[-1],
                "Strategy": episode_strategies,
                "Total rewards": round(episode_reward, 3),
                "State sequence": episode_state_sequence
            }

        return rewards, out_dict  # 返回所有任务的奖励及每一步的状态值

    # 选择动作的方法
    def select_action(self, state):
        """
        根据当前状态选择动作。

        参数:
        - state: 当前状态
        """
        state = torch.tensor(state, dtype=torch.float32)  # 将状态转为张量
        probs = self.actor(state)  # 获取动作概率分布
        action = np.random.choice(self.action_dim, p=probs.cpu().detach().numpy())  # 根据概率选择动作

        return action

    # 模型保存方法
    def save_models(self, path):
        """
        保存Actor和Critic网络的模型参数。

        参数:
        - path: 保存路径
        """
        torch.save(self.actor.state_dict(), f"{path}/actor.pth")
        torch.save(self.critic.state_dict(), f"{path}/critic.pth")

    # 模型加载方法
    def load_models(self, path):
        """
        加载Actor和Critic网络的模型参数。

        参数:
        - path: 加载路径
        """
        self.actor.load_state_dict(torch.load(f"{path}/actor.pth"))
        self.critic.load_state_dict(torch.load(f"{path}/critic.pth"))


# 主函数入口
if __name__ == '__main__':
    # 创建异常检测环境实例
    env = AnomalyMetricEnv(normal_value=0, max_deviation=50, stability=2)
    # 初始化代理
    agent = MetaActorCriticAgent(env)

    # 生成随机数据
    low = -50.0
    high = 50.0
    num_samples = 1000

    nums = np.random.uniform(low, high, num_samples)  # 生成随机数
    nums = np.array(nums)  # 转换为NumPy数组

    # 划分训练集和测试集
    train_data = np.concatenate((nums[:700], nums[800:]))
    train_sort = np.sort(train_data)
    test_data = nums[700:800]

    # 使用MAML进行训练
    agent.maml_train(train_sort)
    # 测试模型
    agent.test(test_data)
    rewards_meta = agent.test(test_data)
    print()
