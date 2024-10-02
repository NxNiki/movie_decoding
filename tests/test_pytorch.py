import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from brain_decoding.param.base_param import device


# Define a simple neural network
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Set device
# device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Load MNIST dataset
train_loader = DataLoader(
    datasets.MNIST("./data", train=True, download=True, transform=transforms.Compose([transforms.ToTensor()])),
    batch_size=64,
    shuffle=True,
)

test_loader = DataLoader(
    datasets.MNIST("./data", train=False, transform=transforms.Compose([transforms.ToTensor()])),
    batch_size=1000,
    shuffle=False,
)

# Initialize the model, loss function, and optimizer
model = SimpleModel().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop (1 epoch for testing)
model.train()
for batch_idx, (data, target) in enumerate(train_loader):
    data, target = data.to(device), target.to(device)

    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()

    if batch_idx % 100 == 0:
        print(f"Train Step: {batch_idx} \tLoss: {loss.item()}")

# Testing loop
model.eval()
test_loss = 0
correct = 0

with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        test_loss += criterion(output, target).item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

test_loss /= len(test_loader.dataset)
print(
    f"\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} "
    f"({100. * correct / len(test_loader.dataset):.0f}%)"
)

# Check if script runs correctly
print("\nPyTorch is correctly installed and working!")
