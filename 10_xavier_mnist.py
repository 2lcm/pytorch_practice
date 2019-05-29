import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import random
import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# data
mnist_train = torchvision.datasets.MNIST("MNIST_data", train=True, transform=torchvision.transforms.ToTensor(), download=True)
mnist_test = torchvision.datasets.MNIST("MNIST_data", train=False, transform=torchvision.transforms.ToTensor(), download=True)

training_epochs = 15
batch_size = 100

dataloader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True, drop_last=True)

linear1 = nn.Linear(28*28, 256, bias=True)
linear2 = nn.Linear(256, 256, bias=True)
linear3 = nn.Linear(256, 10, bias=True)
relu = torch.nn.ReLU()

torch.nn.init.xavier_uniform_(linear1.weight)
torch.nn.init.xavier_uniform_(linear2.weight)
torch.nn.init.xavier_uniform_(linear3.weight)

model = nn.Sequential(linear1, relu, linear2, relu, linear3).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()
total_batch = len(dataloader)
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
    X_test = mnist_test.test_data.view(-1, 28*28).float().to(device)
    Y_test = mnist_test.test_labels.to(device)
    predicted = model(X_test)
    correct_prediction = torch.argmax(predicted, dim=1) == Y_test
    accuracy = correct_prediction.float().mean().item()
    print("Accuracy: ", accuracy)

    # Get one and predict
    r = random.randint(0, len(mnist_test) - 1)
    X_single_data = mnist_test.test_data[r:r + 1].view(-1, 28 * 28).float().to(device)
    Y_single_data = mnist_test.test_labels[r:r + 1].to(device)

    print('Label: ', Y_single_data.item())
    single_prediction = model(X_single_data)
    print('Prediction: ', torch.argmax(single_prediction, 1).item())

    plt.imshow(mnist_test.test_data[r:r + 1].view(28, 28), cmap='Greys', interpolation='nearest')
    plt.show()