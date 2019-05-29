import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F

z = torch.FloatTensor([1, 2, 3])

hypothesis = F.softmax(z, dim=0)
print(hypothesis)
print(hypothesis.sum())

# cross entropy
z = torch.rand(3, 5, requires_grad=True)
hypothesis = F.softmax(z, dim=1)
print(hypothesis)
y = torch.randint(5, (3, )).long()
print(y)
y_one_hot = torch.zeros_like(hypothesis)
print(y.unsqueeze(1))
# scatter_(dim, index, value)
y_one_hot.scatter_(1, y.unsqueeze(1), 1)
print(y_one_hot)
cost = (y_one_hot * -torch.log(hypothesis)).sum(dim=1).mean()
print(cost)
cost = F.nll_loss(F.log_softmax(z, dim=1), y)
print(cost)
cost = F.cross_entropy(z, y)
print(cost)

# Implementation
x_train = [[1, 2, 1, 1],
           [2, 1, 3, 2],
           [3, 1, 3, 4],
           [4, 1, 5, 5],
           [1, 7, 5, 5],
           [1, 2, 5, 6],
           [1, 6, 6, 6],
           [1, 7, 7, 7]]
y_train = [2, 2, 2, 1, 1, 1, 0, 0]
x_train = torch.FloatTensor(x_train)
y_train = torch.LongTensor(y_train)

W = torch.zeros((4, 3), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

n_epochs = 1000
optimizer = optim.SGD([W, b], lr=0.1)
for epoch in range(n_epochs+1):

    z = x_train @ W + b
    cost = F.cross_entropy(z, y_train)

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print('Epoch {:4d}/{} Cost: {:.6f}'.format(
            epoch, n_epochs, cost.item()
        ))
print("#"*40)
class SoftmaxClassifierModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 3)

    def forward(self, x):
        return self.linear(x)

model = SoftmaxClassifierModel()
optimizer = optim.SGD(model.parameters(), lr=0.1)

n_epochs = 1000
for epoch in range(n_epochs + 1):
    prediction = model(x_train)
    cost = F.cross_entropy(prediction, y_train)

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print('Epoch {:4d}/{} Cost: {:.6f}'.format(
            epoch, n_epochs, cost.item()
        ))