import torch
import torchvision as tv

from graph import GCN
from utils import image_to_adj
from tqdm import tqdm

torch.autograd.set_detect_anomaly(True)

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data = tv.datasets.MNIST(root='./data', train=True, download=True)
    targets = data.targets.to(device)
    data = data.data.to(device)
    # downsample the images to 8x8
    train_data = tv.transforms.Resize(8, antialias=True)(data.data)
    train_labels = targets

    adj = image_to_adj(train_data[0].float()) # 64x64 adjacency matrix same for all images

    model = GCN(1, 16, 10, 0.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    if torch.cuda.is_available():
        model = model.cuda()
        adj = adj.cuda()

    loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(10):
        for i, (image, label) in tqdm(enumerate(zip(train_data, train_labels))):
            image = image.float().reshape(-1, 1)
            label = torch.tensor([label], device=device) # shape (1,)
            output = model(image, adj) # shape (10,)
            loss = loss_fn(output, label)
            optimizer.zero_grad()
            optimizer.step()
            loss.backward()

        print(f'Epoch {epoch}, loss: {loss}')
    print('Done!')

    print('Test:')
    test_data = tv.datasets.MNIST(root='./data', train=False, download=True)
    test_labels = test_data.targets.to(device)
    test_data = tv.transforms.Resize(8, antialias=True)(test_data.data).to(device)

    correct = 0
    for i, (image, label) in enumerate(zip(test_data, test_labels)):
        image = image.float().view(-1)
        label = torch.tensor([label], device=device)
        output = model.forward(image, adj)
        if torch.argmax(output) == label:
            correct += 1
    print(f'Accuracy: {correct / len(test_data)}')
    