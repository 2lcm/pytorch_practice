import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# data
x_train = torch.FloatTensor([[1, 2, 1],
                             [1, 3, 2],
                             [1, 3, 4],
                             [1, 5, 5],
                             [1, 7, 5],
                             [1, 2, 5],
                             [1, 6, 6],
                             [1, 7, 7]
                            ])
y_train = torch.LongTensor([2, 2, 2, 1, 1, 1, 0, 0])
x_test = torch.FloatTensor([[2, 1, 1], [3, 1, 2], [3, 3, 4]])
y_test = torch.LongTensor([2, 2, 2])

# class
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 3)
    def forward(self, x):
        return  self.linear(x)

model = MyModel()
optimizer = optim.SGD(model.parameters(), lr=0.1)
def train(model, optimizer, x_train, y_train):
    n_epochs = 1000
    for epoch in range(n_epochs+1):
        prediction = model(x_train)
        cost = F.cross_entropy(prediction, y_train)

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print('Epoch {:4d}/{} Cost: {:.6f}'.format(
                epoch, n_epochs, cost.item()
            ))
    return model

def test(model, optimizer, x_test, y_test):
    prediction = model(x_test)
    # torch.Tensor.max(dim) : (max_value, max_value_index)
    prediction_class = prediction.max(1)[1]
    correct_count = (prediction_class == y_test).sum().item()
    # print(len(x_test))
    # print(x_test.size())
    print(correct_count / y_test.size()[0] * 100)

model = train(model, optimizer, x_train, y_train)
test(model, optimizer, x_test, y_test)


