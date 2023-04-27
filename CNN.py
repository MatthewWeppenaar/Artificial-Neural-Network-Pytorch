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
print(trainset)
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

print(example_data.shape)
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
        self.conv1 = nn.Conv2d(1, 5, 3, padding=1) # First Conv Layer
        self.pool = nn.MaxPool2d(2)  # For pooling
        self.flatten = nn.Flatten() # For flattening the 2D image
        self.fc1 = nn.Linear(980, 256)  # First FC HL
        self.fc2= nn.Linear(256, 10) # Output layer

    def forward(self, x):
      # Batch x of shape (B, C, W, H)
      x = F.relu(self.conv1(x)) # Shape: (B, 5, 28, 28)
      x = self.pool(x)  # Shape: (B, 5, 14, 14)
      x = self.flatten(x) # Shape: (B, 980)
      x = F.relu(self.fc1(x))  # Shape (B, 256)
      x = self.fc2(x)  # Shape: (B, 10)
      return x  
    
cnn = CNN().to(device)

LEARNING_RATE = 1e-1
MOMENTUM = 0.9

# Define the loss function, optimizer, and learning rate scheduler
criterion = nn.CrossEntropyLoss() # Use this if not using softmax layer
optimizer = optim.SGD(cnn.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)

# Train the MLP for 5 epochs
for epoch in range(5):
    train_loss = train(cnn, train_loader, criterion, optimizer, device)
    test_acc = test(cnn, test_loader, device)
    print(f"Epoch {epoch+1}: Train loss = {train_loss:.4f}, Test accuracy = {test_acc:.4f}")
# Test on a batch of data

with torch.no_grad():  # Don't accumlate gradients
  cnn.eval()  # We are in evalutation mode (i.e No dropout)
  x = [data for data in test_loader][0][0].to(device)
  outputs = cnn(x)  # Alias for cnn.forward

  # Print example output.
  print(outputs[0])
  print(f'Prediction: {torch.max(outputs, 1)[1][0]}')