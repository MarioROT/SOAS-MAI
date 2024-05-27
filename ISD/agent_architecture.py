import torch
import torch.nn as nn
import torch.nn.functional as F

class AgentNetwork(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(AgentNetwork, self).__init__()

        # Visual encoder
        self.conv = nn.Conv2d(in_channels=input_shape[0], out_channels=6, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(6 * (input_shape[1] - 2) * (input_shape[2] - 2), 32)
        self.fc2 = nn.Linear(32, 32)

        # LSTM
        self.lstm = nn.LSTMCell(32 + 3, 128)  # input size + extrinsic reward + intrinsic reward + last action

        # Value heads
        self.value_head_extrinsic = nn.Linear(128, 1)
        self.value_head_intrinsic = nn.Linear(128, 1)

        # Policy head
        self.policy_head = nn.Linear(128, num_actions)

        # Intrinsic reward network (evolved)
        self.reward_network = nn.Sequential(
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x, last_action, last_extrinsic_reward, last_intrinsic_reward, hx, cx):
        # Visual encoder
        x = F.relu(self.conv(x))
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        # Ensure last_action, last_extrinsic_reward, and last_intrinsic_reward are 2D tensors
        last_action = last_action.view(-1, 1)
        last_extrinsic_reward = last_extrinsic_reward.view(-1, 1)
        last_intrinsic_reward = last_intrinsic_reward.view(-1, 1)

        # Combine with last action and rewards
        lstm_input = torch.cat([x, last_action, last_extrinsic_reward, last_intrinsic_reward], dim=1)

        # LSTM
        hx, cx = self.lstm(lstm_input, (hx, cx))

        # Value heads
        value_extrinsic = self.value_head_extrinsic(hx)
        value_intrinsic = self.value_head_intrinsic(hx)

        # Policy head
        policy = self.policy_head(hx)

        return policy, value_extrinsic, value_intrinsic, hx, cx

    def get_intrinsic_reward(self, x):
        # Ensure input has 3 channels
        if len(x.shape) == 3 and x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)  # Repeat the single channel to create 3 channels

        x = F.relu(self.conv(x))
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        intrinsic_reward = self.reward_network(x)
        return intrinsic_reward

# Example usage
input_shape = (3, 15, 15)  # 3 channels, 15x15 observation window
num_actions = 8  # Number of actions in the environment

agent = AgentNetwork(input_shape, num_actions)

# Dummy input for testing
x = torch.randn(1, 3, 15, 15)  # Example observation
last_action = torch.randn(1, 1)  # Example last action
last_extrinsic_reward = torch.randn(1, 1)  # Example last extrinsic reward
last_intrinsic_reward = torch.randn(1, 1)  # Example last intrinsic reward
hx = torch.zeros(1, 128)  # Initial hidden state of LSTM
cx = torch.zeros(1, 128)  # Initial cell state of LSTM

policy, value_extrinsic, value_intrinsic, hx, cx = agent(x, last_action, last_extrinsic_reward, last_intrinsic_reward, hx, cx)
intrinsic_reward = agent.get_intrinsic_reward(x)

print("Policy:", policy)
print("Extrinsic Value:", value_extrinsic)
print("Intrinsic Value:", value_intrinsic)
print("Intrinsic Reward:", intrinsic_reward)
print("New hidden state:", hx)
print("New cell state:", cx)
