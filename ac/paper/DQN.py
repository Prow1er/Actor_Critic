import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from newenv import AnomalyMetricEnv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class DQNAgent:
    def __init__(self, env, lr=0.001, gamma=0.98, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01, memory_size=10000,
                 batch_size=64):
        self.env = env
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.memory = []
        self.memory_size = memory_size
        self.batch_size = batch_size

        self.q_network = QNetwork(self.state_dim, self.action_dim).to(device)
        self.target_network = QNetwork(self.state_dim, self.action_dim).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.lr)

    def remember(self, state, action, reward, next_state, done):
        if len(self.memory) > self.memory_size:
            self.memory.pop(0)
        self.memory.append((state, action, reward, next_state, done))

    def select_action(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_dim)
        state = torch.tensor(state, dtype=torch.float32, device=device)
        with torch.no_grad():
            q_values = self.q_network(state)
        return torch.argmax(q_values).item()

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        minibatch = np.random.choice(len(self.memory), self.batch_size, replace=False)
        for idx in minibatch:
            state, action, reward, next_state, done = self.memory[idx]
            state = torch.tensor(state, dtype=torch.float32, device=device)
            next_state = torch.tensor(next_state, dtype=torch.float32, device=device)
            reward = torch.tensor(reward, dtype=torch.float32, device=device)
            done = torch.tensor(done, dtype=torch.float32, device=device)

            target = reward
            if not done:
                target += self.gamma * torch.max(self.target_network(next_state)).item()

            q_values = self.q_network(state)
            q_value = q_values[action]

            loss = (target - q_value).pow(2).mean()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def train(self, val, update_target_every=10):
        num_episodes = len(val)
        initial_states = []
        strategy_sequences = []
        final_states = []

        for episode in range(num_episodes):
            state = self.env.reset(val[episode])
            initial_states.append(state)
            episode_strategies = []

            while True:
                action = self.select_action(state)
                next_state, reward, done, _ = self.env.step(action)

                self.remember(state, action, reward, next_state, done)
                self.replay()

                state = next_state
                episode_strategies.append(self.env.selected_strategies[-1])

                if done:
                    final_states.append(next_state)
                    strategy_sequences.append(episode_strategies)
                    break

            if episode % update_target_every == 0:
                self.update_target_network()

    def test(self, val):
        num_episodes = len(val)
        initial_states = []
        strategy_sequences = []
        final_states = []
        rewards = []
        states_sequences = []  # 新增列表用于保存每个回合的状态序列

        for episode in range(num_episodes):
            state = self.env.reset(val[episode])
            initial_states.append(state)
            episode_strategies = []
            episode_reward = 0
            episode_states = [state.item()]  # 记录本回合的初始状态

            while True:
                action = self.select_action(state)
                next_state, reward, done, _ = self.env.step(action)
                episode_reward += reward

                state = next_state
                episode_strategies.append(self.env.selected_strategies[-1])
                episode_states.append(next_state.item())  # 记录每一步的状态

                if done:
                    final_states.append(next_state)
                    strategy_sequences.append(episode_strategies)
                    rewards.append(episode_reward)
                    states_sequences.append(episode_states)  # 将本回合的所有状态保存

                    # 输出当前回合的信息
                    print(f"Episode {episode + 1}, "
                          f"初始值: {initial_states[-1][0]}, "
                          f"最终值: {final_states[-1][0]}, "
                          f"策略: {episode_strategies}, "
                          f"总奖励值: {episode_reward:.2f}, "
                          f"状态序列: {episode_states}")  # 打印每一步的状态
                    break

        return rewards

    def save_models(self, path):
        torch.save(self.q_network.state_dict(), f"{path}/dqn.pth")

    def load_models(self, path):
        self.q_network.load_state_dict(torch.load(f"{path}/dqn.pth"))



