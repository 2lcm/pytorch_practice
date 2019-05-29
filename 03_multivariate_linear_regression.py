import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
# import torchvision

# data
x_train = torch.FloatTensor([[73, 80, 75], [93, 88, 93], [89, 91, 80], [96, 98, 100], [73, 66, 70]])
y_train = torch.FloatTensor([[152], [185], [180], [196], [142]])

# make model
class MultivariateLinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 1)

    def forward(self, x):
        return self.linear(x)

model = MultivariateLinearRegressionModel()
optimizer = optim.SGD(model.parameters(), lr=1e-5)



n_epochs = 1000
for epoch in range(n_epochs):
    prediction = model(x_train)
    cost = F.mse_loss(prediction, y_train)

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(cost.item())