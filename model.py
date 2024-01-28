import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# convolution neural network
class ConvNeuralNet(nn.Module):
	#  Determine what layers and their order in CNN object 
    def __init__(self):
        super(ConvNeuralNet, self).__init__()
        self.conv_layer1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3)
        self.conv_layer2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3)
        self.max_pool1 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        
        self.conv_layer3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.conv_layer4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)
        self.max_pool2 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        
        self.fc1 = nn.Linear(1600, 128)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 4)
    
    # Progresses data across layers    
    def forward(self, x):
        out = self.conv_layer1(x)
        out = self.conv_layer2(out)
        out = self.max_pool1(out)
        
        out = self.conv_layer3(out)
        out = self.conv_layer4(out)
        out = self.max_pool2(out)
                
        out = out.reshape(out.size(0), -1)
        
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        return out

# load training and test set
def load_data(data_dir):
    # Use transforms.compose method to reformat images into 32x32 size
    all_transforms = transforms.Compose([
        transforms.Resize((32,32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
    ])

    # load dataset
    dataset = ImageFolder(root = data_dir, transform = all_transforms,)

    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    num_examples = len(dataset)

    return dataloader, num_examples

# train the network on the training set
def train(net, trainloader, epochs):
    print(f"Training on {len(trainloader.dataset)} images for {epochs} epochs.....")
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    loss_history = []
    acc_history = []
    for epoch in range(epochs):
        epoch_loss = []
        epoch_acc = []
        for images, labels in trainloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            # Forward pass
            outputs = net(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # log training loss & accuracy
            epoch_loss.append(loss.item())
            _, predicted = torch.max(outputs.data, 1)
            total = labels.size(0)
            correct = (predicted == labels).sum().item()
            epoch_acc.append(100 * correct / total)
        
        # calculate epoch loss & acc
        avg_loss = sum(epoch_loss) / len(epoch_loss)
        avg_acc = sum(epoch_acc) / len(epoch_acc)
        loss_history.append(avg_loss)
        acc_history.append(avg_acc)

        print('Epoch [{}/{}], Training Loss: {:.4f}, Training Acc: {:.2f}%'.format(epoch+1, epochs, avg_loss, avg_acc))
    
    avg_loss=sum(loss_history)/len(loss_history)
    avg_acc=sum(acc_history)/len(acc_history)
    print(f"Training completed, Avg Training Loss: {avg_loss:.2f}, Avg Training Acc: {avg_acc:.2f}%.")

    return avg_loss, avg_acc

# validate the network on the entire test set
def test(net, testloader):
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(DEVICE), data[1].to(DEVICE)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f"Testing completed, Test Acc: {accuracy:.2f}%.")
    return loss, accuracy

