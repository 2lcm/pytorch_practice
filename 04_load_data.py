import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

class CustomDataset(Dataset):
    def __init__(self):
        self.x_train = torch.FloatTensor([[73, 80, 75], [93, 88, 93], [89, 91, 80], [96, 98, 100], [73, 66, 70]])
        self.y_train = torch.FloatTensor([[152], [185], [180], [196], [142]])

    def __len__(self):
        return len(self.x_train)

    def __getitem__(self, idx):
        x = torch.FloatTensor(self.x_train[idx])
        y = torch.FloatTensor(self.y_train[idx])
        return x, y

# make model
class MultivariateLinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 1)

    def forward(self, x):
        return self.linear(x)

model = MultivariateLinearRegressionModel()
optimizer = optim.SGD(model.parameters(), lr=1e-5)

dataset = CustomDataset()
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
n_epochs = 1000

for epoch in range(n_epochs):
    for batch_idx, samples in enumerate(dataloader):
        x_train, y_train = samples
        # print(x_train)
        # print(y_train)

        prediction = model(x_train)
        cost = F.mse_loss(prediction, y_train)

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(cost.item())