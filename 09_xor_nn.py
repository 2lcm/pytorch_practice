import torch
import torch.nn as nn
import torch.optim as optim

device = 'cuda' if torch.cuda.is_available() else 'cpu'

X = torch.FloatTensor([[0, 0], [0, 1], [1, 0], [1, 1]]).to(device)
Y = torch.FloatTensor([[0], [1], [1], [0]]).to(device)

def nn1():
    linear = nn.Linear(2, 1, bias=True)
    sigmoid = nn.Sigmoid()

    model = nn.Sequential(linear, sigmoid).to(device)

    # BCELoss - Binary Cross Entorpy Loss
    criterion = torch.nn.BCELoss().to(device)
    optimizer = optim.SGD(model.parameters(), lr=1)

    n_epochs = 1000
    for step in range(n_epochs+1):
        prediction = model(X)
        cost = criterion(prediction, Y)

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        if step % 100 == 0:
            print(step, cost.item())

    with torch.no_grad():
        hypothesis = model(X)
        predicted = (hypothesis > 0.5).float()
        accuracy = (predicted == Y).float().mean()
        print('\nHypothesis: ', hypothesis.detach().cpu().numpy(), '\nCorrect: ', predicted.detach().cpu().numpy(),
              '\nAccuracy: ', accuracy.item())

def nn2():
    linear1 = nn.Linear(2, 2, bias=True)
    linear2 = nn.Linear(2, 1, bias=True)
    sigmoid = nn.Sigmoid()

    model = nn.Sequential(linear1, sigmoid, linear2, sigmoid).to(device)

    # BCELoss - Binary Cross Entorpy Loss
    criterion = torch.nn.BCELoss().to(device)
    optimizer = optim.SGD(model.parameters(), lr=1)

    n_epochs = 10000
    for step in range(n_epochs+1):
        prediction = model(X)
        cost = criterion(prediction, Y)

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        if step % 1000 == 0:
            print(step, cost.item())

    with torch.no_grad():
        hypothesis = model(X)
        predicted = (hypothesis > 0.5).float()
        accuracy = (predicted == Y).float().mean()
        print('\nHypothesis: ', hypothesis.detach().cpu().numpy(), '\nCorrect: ', predicted.detach().cpu().numpy(),
              '\nAccuracy: ', accuracy.item())

if __name__ == "__main__":
    # nn1()
    nn2()