from __future__ import print_function
import os

import torchvision
import torch.autograd
import torchvision.transforms as transforms
import torch.nn as nn
from vgg import VggNet
from barbar import Bar
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import cv2
from command_line_parser import parse_args
from utilities import create_train_net

args = parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if args.cpu:
    device = 'cpu'

transform = transforms.Compose(
    [transforms.ToTensor(),
     #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
     ])

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

if args.train:
    if __name__ == '__main__':
        start_time = datetime.now()
        print('Started at:', start_time.strftime('%Y-%m-%d %H:%M:%S'))
        print(device)
        if torch.cuda.is_available():
            print('Cuda device count:', torch.cuda.device_count())

        batch_size = 4
        epochs = 3
        learn_rate = 0.001
        if args.learn_rate:
            if args.learn_rate.replace('.', '', 1).isdigit():
                learn_rate = float(args.learn_rate)
        if args.epochs:
            epochs = args.epochs
        if args.train and not os.path.isfile(args.train):
            if not args.vgg_type:
                print("-N/--vgg-type is required when training without an existing checkpoint")
                exit()

        vgg_type = args.vgg_type
        start_epoch = 0

        competition_size = 1
        if args.competition_size:
            competition_size = args.competition_size

        results = []
        print("Training", competition_size, "models, the best one will be saved.")
        for i in range(competition_size):
            # if we've already used this file and it is partially trained, then lets continue
            if os.path.isfile(args.train):
                input_file = torch.load(args.train)
                vgg_type = input_file['vgg_type']
                batch_size = input_file['batch_size']
                start_epoch = input_file['epoch']
                learn_rate = input_file['learn_rate']
                net, optimizer = create_train_net(vgg_type, device, input_file['state_dict'], input_file['optimizer'],
                                                  learn_rate=learn_rate, num_classes=10, size=32)
                if args.batch_size:
                    print('Batch size for this model has already been set to ', batch_size, 'and will not be changed.')
            else:
                start_epoch = 0
                net, optimizer = create_train_net(args.vgg_type, device, learn_rate=learn_rate, num_classes=10, size=32)
                if args.batch_size:
                    batch_size = args.batch_size

            trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                    download=True, transform=transform)

            trainset_indeces = list(range(0, 42500))
            validationset_indeces = list(range(42500, 50000))

            trainset_smaller = torch.utils.data.Subset(trainset, trainset_indeces)
            validation_set = torch.utils.data.Subset(trainset, validationset_indeces)

            trainloader = torch.utils.data.DataLoader(trainset_smaller, batch_size=batch_size,
                                                      shuffle=True, num_workers=4, pin_memory=True)

            validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=batch_size,
                                                      shuffle=True, num_workers=4, pin_memory=True)

            criterion = nn.CrossEntropyLoss()

            net.train()

            epoch_losses = []
            running_loss = 0.0
            for epoch in range(epochs):  # loop over the dataset multiple times
                print('started epoch', start_epoch + epoch + 1, 'of', start_epoch + epochs)
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
                epoch_losses.append(running_loss / len(trainloader.dataset))
                print('Average loss:', running_loss / len(trainloader.dataset))

            # we save the running loss of the last epoch
            output_data = {
                'vgg_type': vgg_type,
                'state_dict': net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': start_epoch + epochs,
                'learn_rate': learn_rate,
                'batch_size': batch_size,
                'loss': running_loss / len(trainloader.dataset),
                'epoch_losses': epoch_losses
            }

            results.append(output_data)

        for i, result in enumerate(results):
            print('Result', i, ' loss:', result['loss'])

        print('Beginning validation')
        best_accuracy = False
        best_output = False
        for result in results:
            net.load_state_dict(result['state_dict'])
            net.eval()
            correct = 0
            total = 0

            for i, data in enumerate(Bar(validation_loader)):
                images, labels = data[0].to(device), data[1].to(device)
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            accuracy = 100 * correct / total
            print('Accuracy of the network on the 7500 validation images: %d %%' % accuracy)
            if not best_accuracy:
                best_accuracy = accuracy
                output = result
            elif best_accuracy < accuracy:
                best_accuracy = accuracy
                output = result

        print('Network with accuracy of ', best_accuracy, 'on validation data set chosen.')
        print('Finished Training')
        end_time = datetime.now()
        print('Completed at:', end_time.strftime('%Y-%m-%d %H:%M:%S'))
        duration = end_time - start_time
        print('Elapsed time', duration.total_seconds(), ' seconds')
        new_dict = {} 
        for k,v in output['state_dict'].items():
            if k.startswith('module.'):
                name = k[7:]
            else:
                name = k
            new_dict[name] = v
        output['state_dict'] = new_dict
        torch.save(output, args.train)
        print('File saved to ', args.train)

if args.execute:
    if __name__ == '__main__':
        if not os.path.isfile(args.execute):
            print('File not found.')
            exit()

        input_file = torch.load(args.execute)

        start_time = datetime.now()
        print('Started at:', start_time.strftime('%Y-%m-%d %H:%M:%S'))
        print(device)
        if torch.cuda.is_available() and device == 'cuda':
            print('Cuda device count:', torch.cuda.device_count())

        batch_size = input_file['batch_size']
        if args.batch_size:
            print('Batch size is only used when training a model.')

        testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                               download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=1,
                                                 shuffle=True, num_workers=4)

        net = VggNet(in_channels=3, num_classes=10, size=32, vgg_type=input_file['vgg_type'], device=device).to(device)
        net.load_state_dict(input_file['state_dict'])
        net.eval()
        correct = 0
        total = 0

        # GRADCAM

        img, _ = next(iter(testloader))

        # get the most likely prediction of the model
        pred = net(img).to(device)
        print(pred)
        predMax = pred.argmax(dim=1)
        print(predMax)

        # get the gradient of the output with respect to the parameters of the model
        pred[:, predMax].backward()
        print('Model thinks this is a', classes[predMax])

        # pull the gradients out of the model
        gradients = net.get_activations_gradient()

        # pool the gradients across the channels
        pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

        # get the activations of the last convolutional layer
        activations = net.get_activations(img).detach()

        # weight the channels by corresponding gradients
        for i in range(512):
            activations[:, i, :, :] *= pooled_gradients[i]

        # average the channels of the activations
        heatmap = torch.mean(activations, dim=1).squeeze()

        # relu on top of the heatmap
        # expression (2) in https://arxiv.org/pdf/1610.02391.pdf
        heatmap = np.maximum(heatmap.cpu(), 0)

        # normalize the heatmap
        heatmap /= torch.max(heatmap)

        # draw the heatmap
        plt.matshow(heatmap.squeeze())
        plt.imsave("./data/heatmap-plot.png", heatmap.squeeze())
        plt.show()

        torchvision.utils.save_image(img[0],'./data/cifar-10-example.png')
        newImage = cv2.imread('./data/cifar-10-example.png')
        heatmap = cv2.resize(np.array(heatmap), (newImage.shape[1], newImage.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        superimposed_img = heatmap * .4 + newImage
        cv2.imwrite('./data/heatmap.jpg', heatmap)
        cv2.imwrite('./data/combined.jpg', superimposed_img)
        exit()
        # /GRADCAM

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
