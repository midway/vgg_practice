from __future__ import print_function
import torchvision
import torch.autograd
import torchvision.transforms as transforms

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


transform = transforms.Compose(
    [transforms.ToTensor(),
     #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
     ])

if __name__ == '__main__':
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)

    trainset_indeces = list(range(0, 42500))
    validationset_indeces = list(range(42500, 50000))

    trainset_smaller = torch.utils.data.Subset(trainset, trainset_indeces)
    validation_set = torch.utils.data.Subset(trainset, validationset_indeces)

    trainloader = torch.utils.data.DataLoader(trainset_smaller, batch_size=50,
                                              shuffle=True, num_workers=4, pin_memory=True)

    validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=50,
                                              shuffle=True, num_workers=4, pin_memory=True)

    for h, data in enumerate(trainloader):
        images, labels = data[0], data[1]
        for i in range(0, 50):
            if labels[i] == 3 or labels[i] == 5:
                if i % 10 == 0:
                    torchvision.utils.save_image(images[i],
                                                 './data/catsdogs/test/' + classes[labels[i].item()] + '/' + str(
                                                     (h * 50) + i) + '.png')
                else:
                    torchvision.utils.save_image(images[i], './data/catsdogs/train/' + classes[labels[i].item()] + '/' + str((h * 50) + i) + '.png')

    for h, data in enumerate(validation_loader):
        images, labels = data[0], data[1]
        for i in range(0, 50):
            if labels[i] == 3 or labels[i] == 5:
                torchvision.utils.save_image(images[i], './data/catsdogs/val/' + classes[labels[i].item()] + '/' + str((h * 50) + i) + '.png')
