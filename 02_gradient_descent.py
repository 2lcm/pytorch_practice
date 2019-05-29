import torch
import torch.optim as optim
# import torchvision

x_train = torch.FloatTensor([[1], [2], [3]])
y_train = 2 * x_train
n_epochs = 1000
lr = 0.01

# Compute gradient descent
W1 = torch.zeros(1)
for epoch in range(n_epochs):
    hypothesis = x_train * W1
    cost = torch.mean((hypothesis - y_train) ** 2)
    gradient = torch.sum((hypothesis - y_train) * x_train)
    if epoch % 100 == 0:
        print("Epoch {:4d}/{} W: {:.3f}, Cost: {:.6f}".format(epoch, n_epochs, W1.item(), cost.item()))

    W1 -= lr * gradient
print("W1: {}".format(W1))

# optimizer method
W2 = torch.zeros(1, requires_grad=True)
optimizer = optim.SGD([W2], lr=lr)
for epoch in range(n_epochs):
    hypothesis = x_train * W2
    cost = torch.mean((hypothesis - y_train) ** 2)

    if epoch % 100 == 0:
        print("Epoch {:4d}/{} W: {:.3f}, Cost: {:.6f}".format(epoch, n_epochs, W2.item(), cost.item()))

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

print("W2: {}".format(W2.detach()))