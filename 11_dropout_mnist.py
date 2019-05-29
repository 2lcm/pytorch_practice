import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim

import random
import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'

mnist_train = torchvision.datasets.MNIST("MNIST_data/", train=True, transform=torchvision.transforms.ToTensor(), download=True)
mnist_test = torchvision.datasets.MNIST("MNIST_data/", train=False, transform=torchvision.transforms.ToTensor(), download=True)

training_epochs = 15
batch_size = 100
learning_rate = 1e-3
drop_prob = 0.3

dataloader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True, drop_last=True)

linear1 = nn.Linear(28*28, 512, bias=True)
linear2 = nn.Linear(512, 512, bias=True)
linear3 = nn.Linear(512, 512, bias=True)
linear4 = nn.Linear(512, 512, bias=True)
linear5 = nn.Linear(512, 10, bias=True)
relu = nn.ReLU()
dropout = nn.Dropout(p=drop_prob)

nn.init.xavier_uniform_(linear1.weight)
nn.init.xavier_uniform_(linear2.weight)
nn.init.xavier_uniform_(linear3.weight)
nn.init.xavier_uniform_(linear4.weight)
nn.init.xavier_uniform_(linear5.weight)

model = nn.Sequential(linear1, relu, dropout,
                      linear2, relu, dropout,
                      linear3, relu, dropout,
                      linear4, relu, dropout, linear5).to(device)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

total_batch = len(dataloader)
model.train()
for epoch in range(training_epochs):
    avg_cost = 0
    for X, Y in dataloader:
        X = X.view(-1, 28*28).float().to(device)
        Y = Y.to(device)

        prediction = model(X)

        optimizer.zero_grad()
        cost = criterion(prediction, Y)
        cost.backward()
        optimizer.step()

        avg_cost += cost / total_batch

    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))
print('Learning finished')

with torch.no_grad():
    model.eval()

    X_test = mnist_test.test_data.view(-1, 28*28).float().to(device)
    Y_test = mnist_test.test_labels.to(device)

    predicted = model(X_test)
    correct_prediction = torch.argmax(predicted, dim=1) == Y_test
    accuracy = correct_prediction.float().mean()
    print("Accuracy: ", accuracy.item())

    r = random.randint(0, len(X_test)-1)
    X_single_data = X_test[r].view(-1, 28*28).float().to(device)
    Y_single_data = Y_test[r].to(device)

    print('Label: ', Y_single_data)
    single_prediction = model(X_single_data)
    print('Prediction: ', torch.argmax(single_prediction, dim=1).item())

    plt.imshow(X_single_data.view(28, 28).cpu().numpy(), cmap="Greys", interpolation='nearest')
    plt.show()