from __future__ import print_function

import itertools
import os
import torch.autograd
import torchvision.transforms as transforms
import torch.nn as nn
from torchvision import datasets
from barbar import Bar
from datetime import datetime

from command_line_parser import parse_args
from vgg import VggNet
import matplotlib.pyplot as plt
import numpy as np
from utilities import create_train_net, print_model_metrics, plot_epoch_losses, print_execution_summary, \
    print_training_summary, save_model

args = parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if args.cpu:
    device = 'cpu'

transforms = {
    'train': transforms.Compose([
        transforms.RandomRotation(5),
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(32, scale=(0.96, 1.0), ratio=(0.95, 1.05)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize([32, 32]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize([32, 32]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

classes = (['dog'])
class_count = len(classes)

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

        epoch_losses = []
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
                epoch_losses = input_file['epoch_losses']
                net, optimizer = create_train_net(vgg_type, device, input_file['state_dict'], input_file['optimizer'],
                                                  learn_rate=learn_rate, num_classes=class_count, size=32)
                if args.batch_size:
                    print('Batch size for this model has already been set to ', batch_size, 'and will not be changed.')
            else:
                start_epoch = 0
                net, optimizer = create_train_net(vgg_type, device, learn_rate=learn_rate, num_classes=class_count, size=32)
                if args.batch_size:
                    batch_size = args.batch_size

            data_dir = 'data/catsdogs'
            image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), transforms[x])
                              for x in ['train', 'val']}
            dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=64,
                                                          shuffle=True, num_workers=4)
                           for x in ['train', 'val']}
            dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
            class_names = image_datasets['train'].classes

            print(class_names)  # => ['cats', 'dogs']
            print(f'Train image size: {dataset_sizes["train"]}')
            print(f'Validation image size: {dataset_sizes["val"]}')


            def imshow(inp, title=None):
                """Imshow for Tensor."""
                inp = inp.numpy().transpose((1, 2, 0))
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                inp = std * inp + mean
                inp = np.clip(inp, 0, 1)
                plt.imshow(inp)
                if title is not None:
                    plt.title(title)
                plt.pause(0.001)  # pause a bit so that plots are updated


            # Get a batch of training data
            # inputs, classes = next(iter(dataloaders['train']))
            # Make a grid from batch
            # sample_train_images = torchvision.utils.make_grid(inputs)
            # imshow(sample_train_images, title=classes)

            criterion = nn.BCEWithLogitsLoss()

            net.train()

            lowest_running_loss = 1000000000
            if len(epoch_losses) > 0:
                lowest_running_loss = min(epoch_losses)
                print('lowest_running_loss:', lowest_running_loss)
            running_loss = 0.0
            running_output_data = None
            for epoch in range(epochs):  # loop over the dataset multiple times
                print('started epoch', start_epoch + epoch + 1, 'of', start_epoch + epochs)
                running_loss = 0.0
                for i, data in enumerate(Bar(dataloaders['train']), 0):
                    # get the inputs; data is a list of [inputs, labels]
                    inputs, labels = data[0].to(device), data[1].to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward + backward + optimize
                    outputs = net(inputs)
                    loss = criterion(outputs, labels.type_as(outputs).unsqueeze(1))
                    loss.backward()
                    optimizer.step()

                    running_loss += float(loss.item())
                epoch_losses.append(running_loss / len(dataloaders['train'].dataset))
                if running_loss < lowest_running_loss:
                    running_output_data = {
                        'vgg_type': vgg_type,
                        'state_dict': net.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'epoch': start_epoch + epoch,
                        'learn_rate': learn_rate,
                        'batch_size': batch_size,
                        'loss': running_loss,
                        'epoch_losses': epoch_losses
                    }
                    lowest_running_loss = running_loss
                    print('Lowest loss epoch found.  Model saved.')

                print('Average loss:', running_loss / len(dataloaders['train'].dataset))

            # we save the running loss of the last epoch
            output_data = {
                'vgg_type': vgg_type,
                'state_dict': net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': start_epoch + epochs,
                'learn_rate': learn_rate,
                'batch_size': batch_size,
                'loss': running_loss,
                'epoch_losses': epoch_losses,
                'best_epoch': running_output_data
            }

            results.append(output_data)

        for i, result in enumerate(results):
            print('Result', i, ' loss:', result['loss'])

        print('Beginning validation')
        best_accuracy = False
        best_output = False
        for result in results:
            net.load_state_dict(result['best_epoch']['state_dict'])
            net.eval()
            correct = 0
            total = 0

            list_of_probabilities = []
            list_of_targets = []
            list_of_predictions = []

            for i, data in enumerate(Bar(dataloaders['val'])):
                images, labels = data[0].to(device), data[1].to(device)
                outputs = net(images)
                probs = outputs.detach()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                list_of_probabilities.append(probs.cpu())
                list_of_predictions.append(predicted.cpu())
                list_of_targets.append(labels.cpu())

            print_model_metrics(list_of_targets, list_of_predictions, list_of_probabilities, prefix='training-')

            accuracy = 100 * correct / total
            print('Accuracy of the network on the validation images: %d %%' % accuracy)
            if not best_accuracy:
                best_accuracy = accuracy
                output = result
            elif best_accuracy < accuracy:
                best_accuracy = accuracy
                output = result

        end_time = datetime.now()
        print_training_summary(best_accuracy, start_time, end_time, output['epoch_losses'])
        new_dict = {}
        for k, v in output['best_epoch']['state_dict'].items():
            if k.startswith('module.'):
                name = k[7:]
            else:
                name = k
            new_dict[name] = v
        output['best_epoch']['state_dict'] = new_dict
        save_model(output['best_epoch'], args.train)


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

        data_dir = 'data/catsdogs'
        image_datasets = datasets.ImageFolder(os.path.join(data_dir, 'test'), transforms['test'])
        dataloader = torch.utils.data.DataLoader(image_datasets, batch_size=64,
                                                 shuffle=True, num_workers=4)

        dataset_sizes = len(image_datasets)
        class_names = image_datasets.classes

        net = VggNet(in_channels=3, num_classes=class_count, size=32, vgg_type=input_file['vgg_type'], device=device).to(device)
        net.load_state_dict(input_file['state_dict'])
        net.eval()
        correct = 0
        total = 0

        #### GRADCAM

        # img, _ = next(iter(testloader))

        # get the most likely prediction of the model
        # pred = net(img).to(device)
        # print(pred)
        # predMax = pred.argmax(dim=1)
        # print(predMax)

        # get the gradient of the output with respect to the parameters of the model
        # pred[:, predMax].backward()
        # print('Model thinks this is a', classes[predMax])

        # pull the gradients out of the model
        # gradients = net.get_activations_gradient()

        # pool the gradients across the channels
        # pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

        # get the activations of the last convolutional layer
        # activations = net.get_activations(img).detach()

        # weight the channels by corresponding gradients
        # for i in range(512):
        #    activations[:, i, :, :] *= pooled_gradients[i]

        # average the channels of the activations
        # heatmap = torch.mean(activations, dim=1).squeeze()

        # relu on top of the heatmap
        # expression (2) in https://arxiv.org/pdf/1610.02391.pdf
        # heatmap = np.maximum(heatmap.cpu(), 0)

        # normalize the heatmap
        # heatmap /= torch.max(heatmap)

        # draw the heatmap
        # plt.matshow(heatmap.squeeze())
        # plt.imsave("./data/heatmap-plot.png", heatmap.squeeze())
        # plt.show()

        # torchvision.utils.save_image(img[0],'./data/cifar-10-example.png')
        # newImage = cv2.imread('./data/cifar-10-example.png')
        # heatmap = cv2.resize(np.array(heatmap), (newImage.shape[1], newImage.shape[0]))
        # heatmap = np.uint8(255 * heatmap)
        # heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        # superimposed_img = heatmap * .4 + newImage
        # cv2.imwrite('./data/heatmap.jpg', heatmap)
        # cv2.imwrite('./data/combined.jpg', superimposed_img)
        # exit()
        #### /GRADCAM

        for i, data in enumerate(Bar(dataloader)):
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Accuracy of the network on the test images: %d %%' % (
                100 * correct / total))

        list_of_probabilities = []
        list_of_targets = []
        list_of_predictions = []

        class_correct = list(0. for i in range(class_count))
        class_total = list(0. for i in range(class_count))
        for i, data in enumerate(Bar(dataloader)):
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            probs = outputs[:, 1].detach()
            c = (predicted == labels).squeeze()
            list_of_predictions.append(predicted.cpu())
            list_of_probabilities.append(probs.cpu())
            list_of_targets.append(labels.cpu())
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

        print_model_metrics(list_of_targets, list_of_predictions, list_of_probabilities, prefix='execution-')

        end_time = datetime.now()
        print_execution_summary(classes, class_correct, class_total, start_time, end_time)
