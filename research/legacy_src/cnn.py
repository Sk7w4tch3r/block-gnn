import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
import torchvision
from torchvision import transforms
import random

random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)

class CNN(nn.Module):
    def __init__(self, f_in, n_classes) -> None:
        super(CNN, self).__init__()
        
        self.in_features = f_in
        self.n_classes = n_classes

        self.conv = nn.Conv2d(f_in, 32, 3, 1)
        self.fc = nn.LazyLinear(n_classes)


    def forward(self, x):
        x = F.relu(self.conv(x))
        x = x.view(-1, x.size(1)*x.size(2)*x.size(3))
        x = self.fc(x)
        return F.log_softmax(x, dim=1)
    
    
    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.n_classes) + ')'

if __name__ == '__main__':

    image_size = (8, 8)

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,)), transforms.Resize(image_size, antialias=True)])
    dataset = torchvision.datasets.MNIST('./data', train=True, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)


    model = CNN(1, 10)
    model.cuda()
    
    criteria = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    for i in range(10):
        for image, label in loader:
            image = image.cuda()
            label = label.cuda()
            optimizer.zero_grad()
            output = model(image)
            loss = criteria(output, label)
            loss.backward()
            optimizer.step()
            

        print(f'loss: {loss.item()}')


    dataset = torchvision.datasets.MNIST('./data', train=False, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for image, label in loader:
            image = image.cuda()
            label = label.cuda()
            output = model(image)
            _, predicted = torch.max(output, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()

    print(f'Accuracy: {correct/total*100}%')


    # save model
    torch.save(model.state_dict(), 'model.pth')