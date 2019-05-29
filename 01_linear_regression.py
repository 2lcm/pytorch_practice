import torch
import torch.optim as optim
# import torchvision

x_train = torch.FloatTensor([[1], [2], [3]])
y_train = 2 * x_train

W = torch.zeros(1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)

optimizer = optim.SGD([W, b], lr=0.01)

n_epochs = 1000
for epoch in range(n_epochs):
    hypothesis = x_train * W + b
    cost = torch.mean((hypothesis - y_train) ** 2)

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

print("W: {}\nb: {}".format(W.item(), b.item()))