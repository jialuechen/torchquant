import torch
import torch.nn as nn
import torch.optim as optim

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64):
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Hyperparameters
state_size = 4
action_size = 2
seed = 0
learning_rate = 0.001

# Initialize Q-network
qnetwork = QNetwork(state_size, action_size, seed)
optimizer = optim.Adam(qnetwork.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

# Example state and target
state = torch.tensor([1.0, 0.0, 0.0, 0.0])
target = torch.tensor([0.0, 1.0])

# Forward pass
output = qnetwork(state)
loss = criterion(output, target)

# Backward pass and optimize
optimizer.zero_grad()
loss.backward()
optimizer.step()

print(f'Q-values: {output}')