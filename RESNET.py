import torch  # Main Package
import torchvision  # Package for Vision Related ML
import torchvision.transforms as transforms  # Subpackage that contains image transforms
from torchvision.transforms.autoaugment import AutoAugmentPolicy
# Create the transform sequence
transform = transforms.Compose([
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




class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1, downsample = None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU())
        self.conv2 = nn.Sequential(
                        nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1),
                        nn.BatchNorm2d(out_channels))
        self.downsample = downsample
        self.relu = nn.ReLU()
        self.out_channels = out_channels
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out



class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes = 10):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Sequential(
                        nn.Conv2d(3, 64, kernel_size = 7, stride = 2, padding = 3),
                        nn.BatchNorm2d(64),
                        nn.ReLU())
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        self.layer0 = self._make_layer(block, 64, layers[0], stride = 1)
        #self.layer1 = self._make_layer(block, 128, layers[1], stride = 2)
        ##self.layer2 = self._make_layer(block, 256, layers[2], stride = 2)
        #self.layer3 = self._make_layer(block, 512, layers[3], stride = 2)
        #self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(4096,128)
        self.fc2 = nn.Linear(128, num_classes)
        
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.layer0(x)
       # x = self.layer1(x)
       # x = self.layer2(x)
       # x = self.layer3(x)

        #x = self.avgpool(x)
        x = x.view(x.size(0), -1)
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

model = ResNet(ResidualBlock, [1, 1, 1, 1]).to(device)

LEARNING_RATE = 1e-3
MOMENTUM = 0.9

# Define the loss function, optimizer, and learning rate scheduler
criterion = nn.CrossEntropyLoss()
''', momentum=MOMENTUM'''
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
lr_decay = optim.lr_scheduler.StepLR(optimizer,5,0.1)
lr_decay2 = optim.lr_scheduler.StepLR(optimizer,8,0.2)
lr_decay3 = optim.lr_scheduler.StepLR(optimizer,15,0.2)

import sys


if sys.argv[1] == "-save":
    test_mlp = []
    test_scores = []    
    for epoch in range(20):
        train_loss = train(model, train_loader, criterion, optimizer, device)
        test_acc = test(model, test_loader, device)
        lr_decay.step()
        lr_decay2.step()
        lr_decay3.step()
        print(f"Epoch {epoch+1}: Train loss = {train_loss:.4f}, Test accuracy = {test_acc:.4f}")
        test_mlp.append(model.state_dict())
        test_scores.append(test_acc)
    


    print("We saving a model")
    
    max_index = test_scores.index(max(test_scores))
    torch.save(test_mlp[max_index], "best_RES.pth")
if sys.argv[1] == '-load':
    res = ResNet(ResidualBlock, [1, 1, 1, 1])
    print("loading params...")
    res.load_state_dict(torch.load("best_RES.pth"))
    print("Done !")

    # Test the loaded model and print the accuracy
    test_acc = test(res, test_loader, device)*100
    print(f"Test accuracy = {test_acc:.2f}%")
