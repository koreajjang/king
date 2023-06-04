import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Data preprocessing and loading (with batch normalization)
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

# Teacher Model (ResNet-18) creation
teacher_model = resnet18(pretrained=True)
teacher_model.fc = nn.Linear(teacher_model.fc.in_features, 10)
teacher_model = teacher_model.to(device)
teacher_model.eval()  # Set to evaluation mode

# Student Model (ResNet-18) creation
student_model = resnet18(num_classes=10)
student_model = student_model.to(device)

# Knowledge distillation parameters
temperature = 3
alpha = 0.5

# Loss function for knowledge distillation
def knowledge_distillation_loss(outputs, labels, teacher_outputs):
    soft_logits = nn.functional.log_softmax(outputs / temperature, dim=1)
    soft_targets = nn.functional.softmax(teacher_outputs / temperature, dim=1)
    return nn.KLDivLoss()(soft_logits, soft_targets) * (temperature ** 2 * alpha) + nn.CrossEntropyLoss()(outputs, labels) * (1. - alpha)


# Optimizer and scheduler
optimizer = optim.SGD(student_model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150, 250], gamma=0.1)

# Training and evaluation
best_accuracy = 0.0
train_losses = []
test_accuracies = []

def train(epoch):
    student_model.train()
    train_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        teacher_outputs = teacher_model(inputs)
        outputs = student_model(inputs)
        loss = knowledge_distillation_loss(outputs, targets, teacher_outputs)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx *
            len(inputs), len(trainloader.dataset),
            100. * batch_idx / len(trainloader), loss.item() / len(inputs)))

    accuracy = 100. * correct / total
    print('Accuracy on the training set: {:.2f}%'.format(accuracy))

    return accuracy


def test():
    student_model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = student_model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    accuracy = 100. * correct / total
    print('Accuracy on the test set: {:.2f}%'.format(accuracy))

    return accuracy


# Training and evaluation
best_accuracy = 0.0
train_losses = []
test_accuracies = []

for epoch in range(300):
    train_accuracy = train(epoch)
    test_accuracy = test()

    train_losses.append(train_accuracy)
    test_accuracies.append(test_accuracy)

    scheduler.step()

    if test_accuracy > best_accuracy:
        best_accuracy = test_accuracy
        torch.save(student_model.state_dict(), 'best_model.pth')

print('Finished training')
print('Best accuracy on the test set: {:.2f}%'.format(best_accuracy))

# Plotting the learning curve
plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
plt.plot(range(1, len(test_accuracies) + 1), test_accuracies, label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.title('Learning Curve')
plt.legend()
plt.show()

