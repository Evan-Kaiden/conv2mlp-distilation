from torchvision import datasets as dsets, transforms
from torch.utils.data import DataLoader

train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_data = dsets.CIFAR100(root='./data', train=True, download=True, transform=train_transform)
test_data = dsets.CIFAR100(root='./data', train=False, download=True, transform=test_transform)

trainloader = DataLoader(train_data, batch_size=64, shuffle=True)
testloader = DataLoader(test_data, batch_size=64, shuffle=False)