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
BATCH_SIZE = 256
train_loader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                          shuffle=True)

test_loader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                          shuffle=False)

#import matplotlib.pyplot as plt

examples = enumerate(test_loader)
batch_idx, (example_data, example_targets) = next(examples)

print(example_data.shape)

import torch
import torch.nn as nn

class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResNetBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += self.shortcut(residual)
        out = self.relu(out)
        
        return out

class CIFAR10ResNet(nn.Module):
    def __init__(self):
        super(CIFAR10ResNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
       # self.resblock1 = ResNetBlock(64, 64, stride=1)
        #self.resblock2 = ResNetBlock(64, 128, stride=2)
        #self.resblock3 = ResNetBlock(128, 256, stride=2)
        self.resblock4 = ResNetBlock(64, 512, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        #x = self.resblock1(x)
        #x = self.resblock2(x)
        # = self.resblock3(x)
        x = self.resblock4(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x


device = ("cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

# Creat the model and send its parameters to the appropriate device
#mlp = ResNet18().to(device)

# Test on a batch of data
#with torch.no_grad():  # Don't accumlate gradients
 # mlp.eval()  # We are in evalutation mode
 # x = example_data.to(device)
  #outputs = mlp(x)  # Alias for mlp.forward

  # Print example output.
  #print(torch.exp(outputs[0]))

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

res = CIFAR10ResNet().to(device)

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
    
    if os.path.exists("best_RES.pth"):
        os.remove("best_RES.pth")

    print("We saving a model")
    
    max_index = test_scores.index(max(test_scores))
    torch.save(test_mlp[max_index], "best_RES.pth")
if sys.argv[1] == '-load':
    res = CIFAR10ResNet()
    print("loading params...")
    res.load_state_dict(torch.load("best_RES.pth"))
    print("Done !")

    # Test the loaded model and print the accuracy
    test_acc = test(res, test_loader, device)*100
    print(f"Test accuracy = {test_acc:.2f}%")
