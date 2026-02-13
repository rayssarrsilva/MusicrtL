import torch
import torch.nn as nn

class RedeSimples(nn.Module):
    def __init__(self):
        super(RedeSimples, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return torch.softmax(self.fc2(x), dim=1)

modelo = RedeSimples()
