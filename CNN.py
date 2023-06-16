import torch  # Main Package
import torchvision  # Package for Vision Related ML
import torchvision.transforms as transforms  # Subpackage that contains image transforms

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
# Test
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                      download=True, transform=transform)

# Send data to the data loaders
BATCH_SIZE = 64
train_loader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                          shuffle=True)

test_loader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                          shuffle=False)

#import matplotlib.pyplot as plt

examples = enumerate(test_loader)
batch_idx, (example_data, example_targets) = next(examples)


# Identify device
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
import torch.nn as nn  # Layers
import torch.nn.functional as F # Activation Functions

# Define the CNN architecture
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
        self.conv1_bn = nn.BatchNorm2d(6)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.conv2_bn = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16,120,kernel_size=5)

        self.flatten = nn.Flatten()
        self.fc2 = nn.Linear(120, 84)
        self.fc1_bn = nn.BatchNorm1d(120)
        self.fc3 = nn.Linear(84, 10)
        #
        self.fc2_bn = nn.BatchNorm1d(84)
     
        self.output = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1_bn(self.conv1(x)))) #14*14*6
        x = self.pool(F.relu(self.conv2_bn(self.conv2(x)))) #16*5*5
        x = F.relu(self.conv3(x)) #1*1*120
        x = self.flatten(x) # flatten all dimensions except batch
        x = self.fc1_bn(x)
        x = F.relu(self.fc2_bn(self.fc2(x)))
        x = self.fc3(x)
        return x
    
cnn = CNN().to(device)

LEARNING_RATE = 1e-3
MOMENTUM = 0.9

# Define the loss function, optimizer, and learning rate scheduler
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(cnn.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
lr_decay = optim.lr_scheduler.StepLR(optimizer,10,0.1) #learning rate decay

import sys 
import os

if sys.argv[1] == "-save": #command line flag for save
    test_mlp = []
    test_scores = []    
    for epoch in range(15):
        train_loss = train(cnn, train_loader, criterion, optimizer, device)
        test_acc = test(cnn, test_loader, device)
        lr_decay.step()
        print(f"Epoch {epoch+1}: Train loss = {train_loss:.4f}, Test accuracy = {test_acc:.4f}")
        test_mlp.append(cnn.state_dict())
        test_scores.append(test_acc)
    

    print("We saving a model")
    
    max_index = test_scores.index(max(test_scores))
    torch.save(test_mlp[max_index], "best_CNN.pth")
if sys.argv[1] == '-load': #command line flag for load
    cnn = CNN().to(device)
    print("loading params...")
    cnn.load_state_dict(torch.load("best_CNN.pth"))
    print("Done !")

    # Test the loaded model and print the accuracy
    test_acc = test(cnn, test_loader, device)*100
    print(f"Test accuracy = {test_acc:.2f}%")
