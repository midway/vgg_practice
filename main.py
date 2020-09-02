from __future__ import print_function
import argparse
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from barbar import Bar
from datetime import datetime
from vgg import VggNet

def check_positive_integer(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError('Only positive integers are allowed.')
    return value


parser = argparse.ArgumentParser(description="Train a new VGG model or use an existing one on the CIFAR-10 data set.")
parser.add_argument('-T', '--train FILE', help='Train a new model and save to file', dest='train')
parser.add_argument('-E', '--epochs X', help='Train the model using X epochs (default: 3)',
                    dest='epochs', type=check_positive_integer)
parser.add_argument('-X', '--execute FILE', help='Execute an existing .pth file on CIFAR-10 data set.', dest='execute')
parser.add_argument('-N', '--vgg-type TYPE', help='VGG type.  Valid values are VGG11 and VGG16.',
                    dest='vgg_type', required=True)
parser.add_argument('-C', '--cpu', help='Force to run only on CPU.', action='store_true')
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if args.cpu:
    device = 'cpu'

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

net = VggNet(in_channels=3, num_classes=10, size=32, vgg_type=args.vgg_type).to(device)
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

if args.train:
    if __name__ == '__main__':
        start_time = datetime.now()
        print('Started at:', start_time.strftime('%Y-%m-%d %H:%M:%S'))
        print(device)
        if torch.cuda.is_available():
            print('Cuda device count:', torch.cuda.device_count())

        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                                  shuffle=True, num_workers=4)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

        epochs = 3
        if args.epochs:
            epochs = args.epochs
        for epoch in range(epochs):  # loop over the dataset multiple times
            print('started epoch', epoch + 1,'of', epochs)
            running_loss = 0.0
            for i, data in enumerate(Bar(trainloader), 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data[0].to(device), data[1].to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += float(loss.item())
            print('Average loss:', running_loss / 10000)
        print('Finished Training')
        torch.save(net.state_dict(), args.train)
        print('File saved to ', args.train)
        end_time = datetime.now()
        print('Completed at:', end_time.strftime('%Y-%m-%d %H:%M:%S'))
        duration = end_time - start_time
        print('Elapsed time', duration.total_seconds(), ' seconds')

if args.execute:
    if __name__ == '__main__':
        start_time = datetime.now()
        print('Started at:', start_time.strftime('%Y-%m-%d %H:%M:%S'))
        print(device)
        if torch.cuda.is_available() and device == 'cuda':
            print('Cuda device count:', torch.cuda.device_count())

        testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                               download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=32,
                                                 shuffle=False, num_workers=4)

        net.load_state_dict(torch.load(args.execute))
        correct = 0
        total = 0
        with torch.no_grad():
            for i, data in enumerate(Bar(testloader)):
                images, labels = data[0].to(device), data[1].to(device)
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print('Accuracy of the network on the 10000 test images: %d %%' % (
            100 * correct / total))

        class_correct = list(0. for i in range(10))
        class_total = list(0. for i in range(10))
        with torch.no_grad():
            for i, data in enumerate(Bar(testloader)):
                images, labels = data[0].to(device), data[1].to(device)
                outputs = net(images)
                _, predicted = torch.max(outputs, 1)
                c = (predicted == labels).squeeze()
                for i in range(4):
                    label = labels[i]
                    class_correct[label] += c[i].item()
                    class_total[label] += 1

        for i in range(10):
            print('Accuracy of %5s : %2d %%' % (
                classes[i], 100 * class_correct[i] / class_total[i]))

        end_time = datetime.now()
        print('Completed at:', end_time.strftime('%Y-%m-%d %H:%M:%S'))
        duration = end_time - start_time
        print('Elapsed time', duration.total_seconds(), ' seconds')
