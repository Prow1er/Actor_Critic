import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import numpy as np
from newenv import AnomalyMetricEnv
from ac_net import Actor  # 注意这里只需要 Actor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PolicyGradientAgent:

    def __init__(self, env, lr=0.001, scheduler_step_size=200, scheduler_gamma=0.5):
        # 初始化环境和模型参数
        self.env = env
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n
        self.actor = Actor(self.state_dim, self.action_dim).to(device)
        self.lr = lr
        # 初始化优化器和学习率调度
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr)
        self.actor_scheduler = StepLR(self.actor_optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma)

    def train(self, val, gamma=0.98, num_training_attempts=1):
        num_episodes = len(val)
        # 初始化存储每个回合的初始状态、策略序列和最终状态的列表
        initial_states = []
        strategy_sequences = []
        final_states = []

        for episode in range(num_episodes):
            # 初始化当前最佳的回合奖励、初始状态、最终状态和策略序列
            best_episode_reward = -float('inf')
            best_initial_state = None
            best_final_state = None
            best_strategy_sequence = None

            for attempt in range(num_training_attempts):
                state = self.env.reset(val[episode])
                episode_reward = 0
                episode_strategies = []
                initial_states.append(state)
                log_probs = []  # 存储每个动作的日志概率
                rewards = []  # 存储每个动作的奖励

                while True:
                    # 选择动作并执行，获取下一个状态和奖励
                    action, log_prob = self.select_action(state)
                    next_state, reward, done, _ = self.env.step(action)
                    episode_reward += reward
                    rewards.append(reward)
                    log_probs.append(log_prob)

                    if done:
                        episode_strategies.append(self.env.selected_strategies[-1])
                        if episode_reward > best_episode_reward:
                            best_episode_reward = episode_reward
                            best_initial_state = initial_states[-1]
                            best_final_state = next_state
                            best_strategy_sequence = episode_strategies[:]
                        break

                    state = next_state
                    episode_strategies.append(self.env.selected_strategies[-1])

                # 计算折扣奖励
                discounted_rewards = self.compute_discounted_rewards(rewards, gamma)
                # 计算策略梯度
                self.update_policy(log_probs, discounted_rewards)

            # 将当前回合的最佳策略序列和最终状态添加到列表中
            strategy_sequences.append(best_strategy_sequence)
            final_states.append(best_final_state)

            # 更新学习率调度器
            self.actor_scheduler.step()

    def test(self, val, gamma=0.98, num_training_attempts=1):
        num_episodes = len(val)
        initial_states = []
        strategy_sequences = []
        final_states = []
        rewards = []

        for episode in range(num_episodes):
            best_episode_reward = -float('inf')
            best_initial_state = None
            best_final_state = None
            best_strategy_sequence = None

            for attempt in range(num_training_attempts):
                state = self.env.reset(val[episode])
                episode_reward = 0
                episode_strategies = []
                initial_states.append(state)

                while True:
                    action, _ = self.select_action(state)  # 测试时不需要计算日志概率
                    next_state, reward, done, _ = self.env.step(action)
                    episode_reward += reward

                    if done:
                        episode_strategies.append(self.env.selected_strategies[-1])
                        if episode_reward > best_episode_reward:
                            best_episode_reward = episode_reward
                            best_initial_state = initial_states[-1]
                            best_final_state = next_state
                            best_strategy_sequence = episode_strategies[:]
                        break

                    state = next_state
                    episode_strategies.append(self.env.selected_strategies[-1])

            strategy_sequences.append(best_strategy_sequence)
            final_states.append(best_final_state)
            rewards.append(best_episode_reward)

            # 打印当前回合的信息
            # print(f"Episode {episode + 1}, "
            #       f"初始值: {best_initial_state[0]}, "
            #       f"最终值: {best_final_state[0]}, "
            #       f"策略: {best_strategy_sequence}, "
            #       f"总奖励值: {best_episode_reward:.2f}")

        return rewards

    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float32, device=device)
        probs = self.actor(state)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob

    def compute_discounted_rewards(self, rewards, gamma):
        discounted_rewards = np.zeros_like(rewards, dtype=np.float32)
        cumulative_reward = 0
        for t in reversed(range(len(rewards))):
            cumulative_reward = rewards[t] + gamma * cumulative_reward
            discounted_rewards[t] = cumulative_reward
        return torch.tensor(discounted_rewards, dtype=torch.float32, device=device)

    def update_policy(self, log_probs, discounted_rewards):
        policy_loss = -torch.sum(torch.stack(log_probs) * discounted_rewards)
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)
        self.actor_optimizer.step()

    def save_models(self, path):
        torch.save(self.actor.state_dict(), f"{path}/pg.pth")

    def load_models(self, path):
        self.actor.load_state_dict(torch.load(f"{path}/pg.pth"))


