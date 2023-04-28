import torch  # Main Package
import torchvision  # Package for Vision Related ML
import torchvision.transforms as transforms  # Subpackage that contains image transforms
from torchvision.transforms.autoaugment import AutoAugmentPolicy
# Create the transform sequence
transform = transforms.Compose([
#transforms.RandomHorizontalFlip(),
#transforms.RandomCrop(32, padding=4),
#transforms.RandomRotation(15),
#transforms.AutoAugment(AutoAugmentPolicy.CIFAR10),
transforms.ToTensor(),  # Convert to Tensor
    # Normalize Image to [-1, 1] first number is mean, second is std deviation
transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

# Load MNIST dataset
# Train
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                      download=True, transform=transform)
print(trainset)
# Test
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                      download=True, transform=transform)

# Send data to the data loaders
BATCH_SIZE = 128
train_loader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                          shuffle=True)

test_loader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                          shuffle=False)

#import matplotlib.pyplot as plt

examples = enumerate(test_loader)
batch_idx, (example_data, example_targets) = next(examples)

print(example_data.shape)

import torch.nn as nn  # Layers
import torch.nn.functional as F # Activation Functions


import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        
    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out



class MyCIFAR10Model(nn.Module):
    def __init__(self):
        super(MyCIFAR10Model, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.res_block1 = ResidualBlock(32, 32)
        #self.res_block2 = ResidualBlock(32, 32)
        self.fc = nn.Linear(32768,32*8*8)
        self.fc2= nn.Linear(32*8*8, 10)
        self.flat = nn.Flatten()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.res_block1(x)
       # x = self.res_block2(x)
        x = self.flat(x)
        x = self.fc(x)
        x = self.fc2(x)
        return x


device = ("cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")


import torch.optim as optim # Optimizers

# Define the training and testing functions
def train(net, train_loader, criterion, optimizer, device):
    net.train()  # Set model to training mode.
    running_loss = 0.0  # To calculate loss across the batches
    for data in train_loader:
        inputs, labels = data  # Get input and labels for batch
        inputs, labels = inputs.to(device), labels.to(device)  # Send to device
        optimizer.zero_grad()  # Zero out the gradients of the ntwork i.e. reset
        outputs = net(inputs)  # Get predictions
        loss = criterion(outputs, labels)  # Calculate loss
        loss.backward()  # Propagate loss backwards
        optimizer.step()  # Update weights
        running_loss += loss.item()  # Update loss
    return running_loss / len(train_loader)
def test(net, test_loader, device):
    net.eval()  # We are in evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():  # Don't accumulate gradients
        for data in test_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device) # Send to device
            outputs = net(inputs)  # Get predictions
            _, predicted = torch.max(outputs.data, 1)  # Get max value
            total += labels.size(0)
            correct += (predicted == labels).sum().item()  # How many are correct?
    return correct / total

res = MyCIFAR10Model().to(device)

LEARNING_RATE = 1e-3
MOMENTUM = 0.9

# Define the loss function, optimizer, and learning rate scheduler
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(res.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
lr_decay = optim.lr_scheduler.StepLR(optimizer,10,0.1)

import sys
import os

if sys.argv[1] == "-save":
    test_mlp = []
    test_scores = []    
    for epoch in range(15):
        train_loss = train(res, train_loader, criterion, optimizer, device)
        test_acc = test(res, test_loader, device)
        lr_decay.step()
        print(f"Epoch {epoch+1}: Train loss = {train_loss:.4f}, Test accuracy = {test_acc:.4f}")
        test_mlp.append(res.state_dict())
        test_scores.append(test_acc)
    


    print("We saving a model")
    
    max_index = test_scores.index(max(test_scores))
    torch.save(test_mlp[max_index], "best_RES.pth")
if sys.argv[1] == '-load':
    res = MyCIFAR10Model()
    print("loading params...")
    res.load_state_dict(torch.load("best_RES.pth"))
    print("Done !")

    # Test the loaded model and print the accuracy
    test_acc = test(res, test_loader, device)*100
    print(f"Test accuracy = {test_acc:.2f}%")


'''import torch  # Main Package
import torchvision  # Package for Vision Related ML
import torchvision.transforms as transforms  # Subpackage that contains image transforms
from torchvision.transforms.autoaugment import AutoAugmentPolicy
# Create the transform sequence
transform = transforms.Compose([
#transforms.RandomHorizontalFlip(),
#transforms.RandomCrop(32, padding=4),
#transforms.RandomRotation(15),
#transforms.AutoAugment(AutoAugmentPolicy.CIFAR10),
transforms.ToTensor(),  # Convert to Tensor
    # Normalize Image to [-1, 1] first number is mean, second is std deviation
transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

# Load MNIST dataset
# Train
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                      download=True, transform=transform)
print(trainset)
# Test
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                      download=True, transform=transform)

# Send data to the data loaders
BATCH_SIZE = 512
train_loader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                          shuffle=True)

test_loader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                          shuffle=False)

#import matplotlib.pyplot as plt

examples = enumerate(test_loader)
batch_idx, (example_data, example_targets) = next(examples)

print(example_data.shape)

import torch.nn as nn  # Layers
import torch.nn.functional as F # Activation Functions


class BasicBlock(nn.Module):
    expansion = 1
    #basic residual block 
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        #self.shortcut = nn.Sequential()
        #if stride != 1 or in_planes != self.expansion*planes:
           # self.shortcut = nn.Sequential(
              #  nn.Conv2d(in_planes, self.expansion*planes,
                         # kernel_size=1, stride=stride, bias=False),
              #  nn.BatchNorm2d(self.expansion*planes)
            #)

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        #out += self.shortcut(x)
        out = F.relu(out)
        out = identity + out
        out = F.relu(out)
        return out
       
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        #innital convolutional layer blocks
        self.layer1 = BasicBlock(64,64)
        #self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        #self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        #self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.flat = nn.Flatten()
        self.linear = nn.Linear(512, num_classes)
    #create a convolutional layer
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        #out = self.layer2(out)
        #out = self.layer3(out)
        #out = self.layer4(out)
        out = self.flat(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

device = ("cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")


import torch.optim as optim # Optimizers

# Define the training and testing functions
def train(net, train_loader, criterion, optimizer, device):
    net.train()  # Set model to training mode.
    running_loss = 0.0  # To calculate loss across the batches
    for data in train_loader:
        inputs, labels = data  # Get input and labels for batch
        inputs, labels = inputs.to(device), labels.to(device)  # Send to device
        optimizer.zero_grad()  # Zero out the gradients of the ntwork i.e. reset
        outputs = net(inputs)  # Get predictions
        loss = criterion(outputs, labels)  # Calculate loss
        loss.backward()  # Propagate loss backwards
        optimizer.step()  # Update weights
        running_loss += loss.item()  # Update loss
    return running_loss / len(train_loader)
def test(net, test_loader, device):
    net.eval()  # We are in evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():  # Don't accumulate gradients
        for data in test_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device) # Send to device
            outputs = net(inputs)  # Get predictions
            _, predicted = torch.max(outputs.data, 1)  # Get max value
            total += labels.size(0)
            correct += (predicted == labels).sum().item()  # How many are correct?
    return correct / total

res = ResNet(BasicBlock, [2, 2, 2, 2])
print(res)

LEARNING_RATE = 1e-3
MOMENTUM = 0.9

# Define the loss function, optimizer, and learning rate scheduler
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(res.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
lr_decay = optim.lr_scheduler.StepLR(optimizer,10,0.1)

import sys
import os

if sys.argv[1] == "-save":
    test_mlp = []
    test_scores = []    
    for epoch in range(15):
        train_loss = train(res, train_loader, criterion, optimizer, device)
        test_acc = test(res, test_loader, device)
        lr_decay.step()
        print(f"Epoch {epoch+1}: Train loss = {train_loss:.4f}, Test accuracy = {test_acc:.4f}")
        test_mlp.append(res.state_dict())
        test_scores.append(test_acc)
    


    print("We saving a model")
    
    max_index = test_scores.index(max(test_scores))
    torch.save(test_mlp[max_index], "best_RES.pth")
if sys.argv[1] == '-load':
    res = ResNet(BasicBlock, [2, 2, 2, 2])
    print("loading params...")
    res.load_state_dict(torch.load("best_RES.pth"))
    print("Done !")

    # Test the loaded model and print the accuracy
    test_acc = test(res, test_loader, device)*100
    print(f"Test accuracy = {test_acc:.2f}%")
'''