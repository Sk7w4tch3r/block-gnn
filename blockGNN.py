import time
import torch
import torch.nn as nn
import torchvision as tv
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, default_collate
import matplotlib.pyplot as plt
import random
from gcn.gnn import BlockSparseCNN


# set random seed
random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

image_size = (8, 8)

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,)), transforms.Resize(image_size, antialias=True)])
dataset = tv.datasets.MNIST('./data', train=True, download=True, transform=transform)
loader = DataLoader(dataset, batch_size=128, shuffle=True)


model = BlockSparseCNN(1, 10)

if torch.cuda.is_available():
    model.cuda()


criteria = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

model = model.cuda()

correct = 0
total = 0
best_loss = 100

for i in range(20):
    start = time.time()
    for image, label in loader:
        image = image.cuda()
        label = label.cuda()
        optimizer.zero_grad()
        output = model(image)
        loss = criteria(output, label)
        loss.backward()
        optimizer.step()
        _, predicted = torch.max(output, 1)
        total += label.size(0)
        correct += (predicted == label).sum().item()
    if loss.item() < best_loss:
        best_loss = loss.item()
        torch.save(model.state_dict(), f'bsparse_{image_size[0]}.pth')
    print(f'loss: {loss.item()}, accuracy: {correct/total*100:.2f}%, time: {time.time()-start:.2f}s')


dataset = tv.datasets.MNIST('./data', train=False, download=True, transform=transform)
loader = DataLoader(dataset, batch_size=128, shuffle=True)

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
plt.imshow(model.conv.coeff.to_dense()[0, 0].detach().cpu().numpy())
plt.show()

# save model
torch.save(model.state_dict(), f'bsparse_{image_size[0]}.pth')