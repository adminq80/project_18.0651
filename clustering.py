#!/usr/bin/env python
# coding: utf-8


import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

import copy
from collections import defaultdict
from random import choice


import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output



def train(model, device, train_loader, optimizer, epoch, verbose=False):
    model.to(device)
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if (batch_idx % 10  == 0) and verbose:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, test_loader):
    model.to(device)
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))



transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
dataset1 = datasets.MNIST('data', train=True, download=True, transform=transform)
dataset2 = datasets.MNIST('data', train=False, transform=transform)



images_with_targets = [i for i in zip(dataset1.data, dataset1.targets)]
images_with_targets[0][1]


data0 = [i for i in filter(lambda x: x[1] == 0, images_with_targets)]
data1 = [i for i in filter(lambda x: x[1] == 1, images_with_targets)]
data2 = [i for i in filter(lambda x: x[1] == 2, images_with_targets)]
data3 = [i for i in filter(lambda x: x[1] == 3, images_with_targets)]
data4 = [i for i in filter(lambda x: x[1] == 4, images_with_targets)]
data5 = [i for i in filter(lambda x: x[1] == 5, images_with_targets)]
data6 = [i for i in filter(lambda x: x[1] == 6, images_with_targets)]
data7 = [i for i in filter(lambda x: x[1] == 7, images_with_targets)]
data8 = [i for i in filter(lambda x: x[1] == 8, images_with_targets)]
data9 = [i for i in filter(lambda x: x[1] == 9, images_with_targets)]


models = [Net() for i in range(10)]


class DataWrapper(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = []
        self.targets = []
        for i, l in data:
            self.data.append(i.type(torch.float))
            self.targets.append(l.type(torch.long))
    def __len__(self):
        return len(self.data)
    def __getitem__(self, item):
        return self.data[item].view((1, 28, 28)), self.targets[item]



data_loader0 = torch.utils.data.DataLoader(DataWrapper(data0), batch_size=64, shuffle=True)
data_loader1 = torch.utils.data.DataLoader(DataWrapper(data1), batch_size=64, shuffle=True)
data_loader2 = torch.utils.data.DataLoader(DataWrapper(data2), batch_size=64, shuffle=True)
data_loader3 = torch.utils.data.DataLoader(DataWrapper(data3), batch_size=64, shuffle=True)
data_loader4 = torch.utils.data.DataLoader(DataWrapper(data4), batch_size=64, shuffle=True)
data_loader5 = torch.utils.data.DataLoader(DataWrapper(data5), batch_size=64, shuffle=True)
data_loader6 = torch.utils.data.DataLoader(DataWrapper(data6), batch_size=64, shuffle=True)
data_loader7 = torch.utils.data.DataLoader(DataWrapper(data7), batch_size=64, shuffle=True)
data_loader8 = torch.utils.data.DataLoader(DataWrapper(data8), batch_size=64, shuffle=True)
data_loader9 = torch.utils.data.DataLoader(DataWrapper(data9), batch_size=64, shuffle=True)
data_loaders = [data_loader0, data_loader1, data_loader2, data_loader3, 
                data_loader4, data_loader5, data_loader6, data_loader7,
               data_loader8, data_loader9]



m_and_d = [i for i in zip(models, data_loaders)]
m_and_d[0]



global_model = Net()
optimizer = optim.Adadelta(global_model.parameters(), lr=1e-4)


def fed_avg(weights):
    w = copy.deepcopy(weights[0]) 
    for k in w.keys():
        for i in range(1, len(weights)):
            w[k] += weights[i][k]
        w[k] = torch.div(w[k], len(weights))
    return w


weights = defaultdict(list)


for i in range(1000):
    print(i) if (i % 100 == 0) else None
    device = 'cuda:{}'.format(choice([0,1,2,3]))
    for m, d in m_and_d:
        m.load_state_dict(copy.deepcopy(global_model.state_dict()))
        train(m, device, d, optimizer, 50)
        # move the model back to cpu and save it into a dict
        weights[i].append(copy.deepcopy(m.cpu().state_dict()))
        # weights[i].append(copy.deepcopy(m.state_dict()))

    global_model.load_state_dict(fed_avg(weights[i]))


for i, client in enumerate(weights[max(weights.keys())]):
    client = client['weight']
    layers = torch.cat((client['conv1.weight'].reshape(1, -1),
                        client['conv1.bias'].reshape(1, -1),
                        client['conv2.weight'].reshape(1, -1),
                        client['conv2.bias'].reshape(1, -1),
                        client['fc1.weight'].reshape(1, -1),
                        client['fc1.bias'].reshape(1, -1),
                        client['fc2.weight'].reshape(1, -1),
                        client['fc2.bias'].reshape(1, -1),
                       ), dim=1)
    if i == 0 :
        clients = layers
        print(layers.size())
        continue
    clients = torch.cat((clients, layers,), dim=0)
print(clients.size())


kmeans = KMeans(n_clusters=10)


kmeans.fit(clients.cpu().numpy())


centers = kmeans.cluster_centers_


y_kmeans = kmeans.predict(clients.cpu().numpy())


X = clients.cpu().numpy()


plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')
plt.savefig('knn_scatter.jpg', dpi=330)


a = np.zeros((10, 2))
for j in range(10):
    a[j, 0] = j
    for i, w in enumerate(weights[49]):
        if w['labels'][0][0] == j:
            a[j, 1] += 1


plt.bar(a[:, 0], a[:, 1])
plt.axhline(y=10, color='r', linestyle='-')
plt.ylabel('Number of agents')
plt.xlabel('Predicted cluster')
plt.savefig('knn_cluster_fed.png', dpi=330)


