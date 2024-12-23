import torch
import torch.optim as optim
import numpy as np
from collections import deque
import random
from newenv import AnomalyMetricEnv
from torch import nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class QLearningAgent:
    def __init__(self, env, lr=0.001, gamma=0.98, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995, batch_size=64, replay_buffer_size=10000):
        self.env = env
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.replay_buffer = deque(maxlen=replay_buffer_size)

        self.q_network = QNetwork(self.state_dim, self.action_dim).to(device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_dim)
        state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        q_values = self.q_network(state_tensor)
        return q_values.argmax().item()

    def add_to_replay_buffer(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))

    def train(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        # Sample a batch of experiences from the replay buffer
        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(states, dtype=torch.float32, device=device)
        actions = torch.tensor(actions, dtype=torch.long, device=device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
        next_states = torch.tensor(next_states, dtype=torch.float32, device=device)
        dones = torch.tensor(dones, dtype=torch.float32, device=device)

        # Compute Q values and target Q values
        q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze()
        with torch.no_grad():
            next_q_values = self.q_network(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        # Compute loss and optimize the Q-network
        loss = self.loss_fn(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def train_agent(self, val):
        num_episodes = len(val)
        for episode in range(num_episodes):
            state = self.env.reset(val[episode])
            episode_reward = 0

            while True:
                action = self.select_action(state)
                next_state, reward, done, _ = self.env.step(action)
                episode_reward += reward

                # Store experience in replay buffer
                self.add_to_replay_buffer(state, action, reward, next_state, done)

                # Train the network
                self.train()

                if done:
                    break

                state = next_state

            # print(f"Episode {episode + 1}, Reward: {episode_reward}")

    def test(self, val):
        num_episodes = len(val)
        self.q_network.eval()  # Set the network to evaluation mode
        results = []

        for episode in range(num_episodes):
            state = self.env.reset(val[episode])
            initial_state = state
            episode_reward = 0
            episode_strategies = []

            while True:
                action = self.select_action(state)
                next_state, reward, done, _ = self.env.step(action)
                episode_reward += reward
                episode_strategies.append(action)

                if done:
                    break

                state = next_state

            # Convert action indices to strategy names
            strategies = self.env.strategy_library.get_strategies(initial_state[0])
            applied_strategies = [strategies[a % len(strategies)].name for a in episode_strategies]

            results.append({
                'initial_state': initial_state[0],
                'final_state': next_state[0],
                'strategies': applied_strategies,
                'total_reward': episode_reward
            })

            # print(f"Episode {episode + 1}, "
            #       f"初始值: {initial_state[0]}, "
            #       f"最终值: {next_state[0]}, "
            #       f"策略: {applied_strategies}, "
            #       f"总奖励: {episode_reward}")

        return results

    def save_models(self, path):
        torch.save(self.q_network.state_dict(), f"{path}/ql.pth")

    def load_models(self, path):
        self.q_network.load_state_dict(torch.load(f"{path}/ql.pth"))



