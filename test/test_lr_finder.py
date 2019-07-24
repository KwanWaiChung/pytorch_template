import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytest
import torchvision
import torchvision.transforms as transforms
from ..utils.lr_finder import LRFinder
from ..utils.logger import getlogger
from .mock_handler import MockLoggingHandler


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class TestLRFinder:
    def setup_method(self, method):
        self.handler = MockLoggingHandler(level="DEBUG")
        logger = getlogger()
        logger.parent.addHandler(self.handler)

        self.model = Net()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        trainset = torchvision.datasets.CIFAR10(
            root=f"{os.path.dirname(os.path.realpath(__file__))}/data",
            train=True,
            download=True,
            transform=transform,
        )
        self.trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=4, shuffle=True, num_workers=2
        )

        testset = torchvision.datasets.CIFAR10(
            root="./data", train=False, download=True, transform=transform
        )
        self.testloader = torch.utils.data.DataLoader(
            testset, batch_size=4, shuffle=False, num_workers=2
        )

    def test_exponential_scheduling(self):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=1e-6)

        lr_finder = LRFinder(
            self.model, optimizer, criterion, device=self.device
        )
        #  lr_finder(self.trainloader, self.testloader, 1, 100)
        lr_finder(self.trainloader, None, 1, 10)
        lr_finder.plot()
        assert (
            self.handler.messages["info"][-1]
            == "Learning rate search finished. "
            "See the graph with {finder_name}.plot()"
        )
