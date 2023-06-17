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


import torch.nn as nn  # Layers
import torch.nn.functional as F # Activation Functions




class SimpleResidualBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
                        nn.Conv2d(128,128, kernel_size = 3, stride = 1, padding = 1),
                        nn.BatchNorm2d(128),
                        nn.ReLU())
        self.conv2 = nn.Sequential(
                        nn.Conv2d(128, 128, kernel_size = 3, stride = 1, padding = 1),
                        nn.BatchNorm2d(128))
        
        self.relu = nn.ReLU()
        
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out += residual
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels,64, kernel_size=3, padding=1)
        self.conv1_bn = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3,padding=1)
        self.conv2_bn = nn.BatchNorm2d(128)

        #creating a simple residual block
        self.res1 = SimpleResidualBlock()

        self.pool2 = nn.MaxPool2d(4)
        self.fc1 = nn.Linear(2048,512)
        self.fc2 = nn.Linear(512,num_classes)
        self.flatten = nn.Flatten()
 
    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.pool(F.relu(out))
        out = self.res1(out)
        out = self.pool2(out)
        out = self.flatten(out)
        out = self.fc1(out)
        out = self.fc2(out)
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

model = ResNet(3, 10).to(device)

LEARNING_RATE = 1e-3
MOMENTUM = 0.9

# Define the loss function, optimizer, and learning rate scheduler
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
lr_decay = optim.lr_scheduler.StepLR(optimizer,10,0.1)


import sys


if sys.argv[1] == "-save":
    test_mlp = []
    test_scores = []    
    for epoch in range(15):
        train_loss = train(model, train_loader, criterion, optimizer, device)
        test_acc = test(model, test_loader, device)
        lr_decay.step()
        print(f"Epoch {epoch+1}: Train loss = {train_loss:.4f}, Test accuracy = {test_acc:.4f}")
        test_mlp.append(model.state_dict())
        test_scores.append(test_acc)
    


    print("We saving a model")
    
    max_index = test_scores.index(max(test_scores))
    torch.save(test_mlp[max_index], "best_RES.pth")
if sys.argv[1] == '-load':
    res = ResNet(3, 10).to(device)
    print("loading params...")
    res.load_state_dict(torch.load("best_RES.pth"))
    print("Done !")

    # Test the loaded model and print the accuracy
    test_acc = test(res, test_loader, device)*100
    print(f"Test accuracy = {test_acc:.2f}%")
