import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import numpy as np
from newenv import AnomalyMetricEnv
from ac_net import Actor, Critic

device = torch.device("cpu")


class ActorCriticAgent:

    def __init__(self, env, lr=0.001, scheduler_step_size=200, scheduler_gamma=0.5):
        # 初始化环境和模型参数
        self.env = env
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n
        self.actor = Actor(self.state_dim, self.action_dim).to(device)
        self.critic = Critic(self.state_dim).to(device)
        self.lr = lr
        # 初始化优化器和学习率调度
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.lr)
        self.actor_scheduler = StepLR(self.actor_optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma)
        self.critic_scheduler = StepLR(self.critic_optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma)

    def train(self, val, gamma=0.98):
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


            state = self.env.reset(val[episode])
            episode_reward = 0
            episode_strategies = []
            initial_states.append(state)

            while True:
                # 选择动作并执行，获取下一个状态和奖励
                action = self.select_action(state)
                next_state, reward, done, _ = self.env.step(action)
                episode_reward += reward

                # 将状态、下一个状态和奖励转换为张量
                state_tensor = torch.tensor(state, dtype=torch.float32, device=device)
                next_state_tensor = torch.tensor(next_state, dtype=torch.float32, device=device)
                reward_tensor = torch.tensor(reward, dtype=torch.float32, device=device)

                # 计算当前状态和下一个状态的价值
                value = self.critic(state_tensor)
                next_value = self.critic(next_state_tensor)

                # 计算目标值
                target = reward_tensor + (1 - done) * gamma * next_value
                target = target.detach()

                # 计算评论家损失并反向传播
                critic_loss = (target - value).pow(2).mean()
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)
                self.critic_optimizer.step()

                # 计算日志概率和演员损失并反向传播
                log_prob = torch.log(self.actor(state_tensor)[action])
                advantage = target - value
                actor_loss = -log_prob * advantage.detach()
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

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

            # 将当前回合的最佳策略序列和最终状态添加到列表中
            strategy_sequences.append(best_strategy_sequence)
            final_states.append(best_final_state)
            # 打印当前回合的信息
            # print(f"Episode {episode + 1}, "
            #       f"初始值: {best_initial_state}, "
            #       f"最终值: {best_final_state}, "
            #       f"策略: {best_strategy_sequence}"
            #       f"总奖励: {best_episode_reward:.2f}")

            # 更新学习率调度器
            self.actor_scheduler.step()
            self.critic_scheduler.step()

    def test(self, val):
        num_episodes = len(val)
        self.actor.eval()
        self.critic.eval()

        # Initialize lists to store test episode information
        initial_states = []
        final_states = []
        strategy_sequences = []
        rewards = []
        states_per_episode = []  # List to hold the states of each episode

        for episode in range(num_episodes):
            state = self.env.reset(val[episode])
            initial_states.append(state)
            episode_reward = 0
            episode_strategies = []
            episode_states = [state.item()]  # Start with the initial state

            while True:
                action = self.select_action(state)
                next_state, reward, done, _ = self.env.step(action)
                episode_reward += reward

                # Record strategy and state
                episode_strategies.append(self.env.selected_strategies[-1])
                episode_states.append(next_state.item())  # Record the state after taking an action

                if done:
                    final_states.append(next_state)
                    break

                state = next_state

            rewards.append(episode_reward)
            strategy_sequences.append(episode_strategies)
            states_per_episode.append(episode_states)  # Store all states of this episode

            # Print episode summary
            print(f"Episode {episode + 1}, "
                  f"初始值: {initial_states[-1]}, "
                  f"最终值: {final_states[-1]}, "
                  f"策略: {episode_strategies}, "
                  f"总奖励: {episode_reward:.2f}, "
                  f"每步状态: {episode_states}")

        # Return collected information
        return rewards

    def select_action(self, state):
        # 将状态转换为张量
        state = torch.tensor(state, dtype=torch.float32, device=device)
        # 计算状态对应的动作概率
        probs = self.actor(state)
        # 根据概率选择动作
        action = np.random.choice(self.action_dim, p=probs.cpu().detach().numpy())
        return action

    def save_models(self, path):
        # 保存actor和critic模型的参数
        torch.save(self.actor.state_dict(), f"{path}/actor.pth")
        torch.save(self.critic.state_dict(), f"{path}/critic.pth")

    def load_models(self, path):
        # 加载actor和critic模型的参数
        self.actor.load_state_dict(torch.load(f"{path}/actor.pth"))
        self.critic.load_state_dict(torch.load(f"{path}/critic.pth"))


if __name__ == '__main__':
    env = AnomalyMetricEnv(normal_value=0, max_deviation=50, stability=2)
    agent = ActorCriticAgent(env)

    low = -50.0
    high = 50.0
    num_samples = 1000

    # 生成随机实数
    nums = np.random.uniform(low, high, num_samples)
    nums = np.array(nums)

    # 训练数据和测试数据
    train_data = np.concatenate((nums[:700], nums[800:]))
    test_data = nums[700:800]
    # agent.train(train_data)
    agent.test(test_data)
    print()
