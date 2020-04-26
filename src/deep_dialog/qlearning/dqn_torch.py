import torch
import torch.nn as nn
import torch.optim as optim
# import torch.nn.functional as F

from torch.autograd import Variable


class DQN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DQN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.linear_i2h = nn.Linear(self.input_size, self.hidden_size)
        self.linear_h2o = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x):
        x = torch.tanh(self.linear_i2h(x))
        x = self.linear_h2o(x)
        return x

    def predict(self, x):
        y = self.forward(x)
        return torch.argmax(y, 1)


class DQN2(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DQN2, self).__init__()

        self.linear1 = nn.Linear(self.input_size, self.hidden_size)
        self.linear2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.linear3 = nn.Linear(self.hidden_size, self.output_size)

        self.linear4 = nn.Linear(self.input_size, self.hidden_size)
        self.linear5 = nn.Linear(self.hidden_size, self.hidden_size)
        self.linear6 = nn.Linear(self.hidden_size, self.output_size)


    def forward(self, x):
        x1 = torch.tanh(self.linear1(x))
        x1 = torch.tanh(self.linear2(x1))
        x1 = self.linear3(x1)

        x2 = torch.tanh(self.linear4(x))
        x2 = torch.tanh(self.linear5(x2))
        x2 = self.linear6(x2)
        return x1, x2

    def predict(self, x):
        y1, y2 = self.forward(x)
        return torch.argmax(torch.min(y1, y2), 1)