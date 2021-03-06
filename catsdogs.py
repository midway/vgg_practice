from __future__ import print_function
import argparse
import itertools
import os
import torchvision
import torch.autograd
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
from barbar import Bar
from datetime import datetime
from vgg import VggNet
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix

def check_positive_integer(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError('Only positive integers are allowed.')
    return value


def results_sorter(value):
    return value['loss']


def plot_roc_curve(fpr, tpr):
    plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.savefig('figure.png')
    plt.show()


parser = argparse.ArgumentParser(description="Train a new VGG model or use an existing one on the CIFAR-10 data set.")
parser.add_argument('-T', '--train FILE',
                    help='Train a new model and save to file (if file exists it will be used to continue training)',
                    dest='train')
parser.add_argument('-E', '--epochs X', help='Train the model using X epochs (default: 3)',
                    dest='epochs', type=check_positive_integer)
parser.add_argument('-X', '--execute FILE', help='Execute an existing .pth file on CIFAR-10 data set.', dest='execute')
parser.add_argument('-N', '--vgg-type TYPE', help='VGG type.  Valid values are VGG11 and VGG16.',dest='vgg_type')
parser.add_argument('-C', '--cpu', help='Force to run only on CPU.', action='store_true')
parser.add_argument('-B', '--batch-size', help='Batch size used for training.  (default: 4)', dest='batch_size')
parser.add_argument('-S', '--competition-size X', help='Train X models and save only the best performing one (least loss)', dest='competition_size', type=check_positive_integer)
args = parser.parse_args()

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
        transforms.Resize([32,32]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize([32, 32]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

classes = ( 'cat', 'dog' )

if args.train:
    if __name__ == '__main__':
        start_time = datetime.now()
        print('Started at:', start_time.strftime('%Y-%m-%d %H:%M:%S'))
        print(device)
        if torch.cuda.is_available():
            print('Cuda device count:', torch.cuda.device_count())

        batch_size = 4
        epochs = 3
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
        print("Training",competition_size,"models, the best one will be saved.")
        for i in range(competition_size):
            # if we've already used this file and it is partially trained, then lets continue
            if os.path.isfile(args.train):
                input_file = torch.load(args.train)
                batch_size = input_file['batch_size']
                net = VggNet(in_channels=3, num_classes=2, size=32, vgg_type=input_file['vgg_type'], device=device).to(device)
                net.load_state_dict(input_file['state_dict'])
                net.train()
                optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
                optimizer.load_state_dict(input_file['optimizer'])
                start_epoch = input_file['epoch']
                if args.batch_size:
                    print('Batch size for this model has already been set to ', batch_size, 'and will not be changed.')
            else:
                start_epoch = 0
                net = VggNet(in_channels=3, num_classes=2, size=32, vgg_type=args.vgg_type, device=device).to(device)
                net.train()
                optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
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
            inputs, classes = next(iter(dataloaders['train']))
            # Make a grid from batch
            sample_train_images = torchvision.utils.make_grid(inputs)
            imshow(sample_train_images, title=classes)

            criterion = nn.CrossEntropyLoss()

            net.train()

            running_loss = 0.0
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
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    running_loss += float(loss.item())
                print('Average loss:', running_loss / 10000)

            # we save the running loss of the last epoch
            output_data = {
                'vgg_type': vgg_type,
                'state_dict': net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': start_epoch + epochs,
                'batch_size': batch_size,
                'loss': running_loss / 10000
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

            list_of_probabilities = []
            list_of_targets = []
            list_of_predictions = []

            for i, data in enumerate(Bar(dataloaders['val'])):
                images, labels = data[0].to(device), data[1].to(device)
                outputs = net(images)
                probs = outputs[:, 1].detach()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                list_of_probabilities.append(probs.cpu())
                list_of_predictions.append(predicted.cpu())
                list_of_targets.append(labels.cpu())

            tn, fp, fn, tp = confusion_matrix(torch.cat(list_of_targets, dim=0).cpu(), torch.cat(list_of_predictions, dim=0).cpu()).ravel()
            print(tn, fp, fn, tp)

            ppv = tp / (tp + fp)
            print('Positive Predictive Value:', ppv)

            npv = tn / (tn + fn)
            print('Negative Predictive Value', npv)

            specificity = tn / (tn + fp)
            print('Specificity:', specificity)

            sensitivity = tp / (tp + fn)
            print('Sensitivity:', sensitivity)


            fpr, tpr, thresholds = roc_curve(torch.cat(list_of_targets, dim=0).cpu(),
                                             torch.cat(list_of_probabilities, dim=0).cpu())
            plot_roc_curve(fpr, tpr)
            auc_score = roc_auc_score(torch.cat(list_of_targets, dim=0).cpu(),
                                      torch.cat(list_of_probabilities, dim=0).cpu())
            print('AUC Score:', auc_score)

            accuracy = 100 * correct / total
            print('Accuracy of the network on the validation images: %d %%' % accuracy)
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

        data_dir = 'data/catsdogs'
        image_datasets = datasets.ImageFolder(os.path.join(data_dir, 'test'), transforms['test'])
        dataloader = torch.utils.data.DataLoader(image_datasets, batch_size=64,
                                                      shuffle=True, num_workers=4)

        dataset_sizes = len(image_datasets)
        class_names = image_datasets.classes

        net = VggNet(in_channels=3, num_classes=2, size=32, vgg_type=input_file['vgg_type'], device=device).to(device)
        net.load_state_dict(input_file['state_dict'])
        net.eval()
        correct = 0
        total = 0

        #### GRADCAM

        #img, _ = next(iter(testloader))

        # get the most likely prediction of the model
        #pred = net(img).to(device)
        #print(pred)
        #predMax = pred.argmax(dim=1)
        #print(predMax)

        # get the gradient of the output with respect to the parameters of the model
        #pred[:, predMax].backward()
        #print('Model thinks this is a', classes[predMax])

        # pull the gradients out of the model
        #gradients = net.get_activations_gradient()

        # pool the gradients across the channels
        #pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

        # get the activations of the last convolutional layer
        #activations = net.get_activations(img).detach()

        # weight the channels by corresponding gradients
        #for i in range(512):
        #    activations[:, i, :, :] *= pooled_gradients[i]

        # average the channels of the activations
        #heatmap = torch.mean(activations, dim=1).squeeze()

        # relu on top of the heatmap
        # expression (2) in https://arxiv.org/pdf/1610.02391.pdf
        #heatmap = np.maximum(heatmap.cpu(), 0)

        # normalize the heatmap
        #heatmap /= torch.max(heatmap)

        # draw the heatmap
        #plt.matshow(heatmap.squeeze())
        #plt.imsave("./data/heatmap-plot.png", heatmap.squeeze())
        #plt.show()

        #torchvision.utils.save_image(img[0],'./data/cifar-10-example.png')
        #newImage = cv2.imread('./data/cifar-10-example.png')
        #heatmap = cv2.resize(np.array(heatmap), (newImage.shape[1], newImage.shape[0]))
        #heatmap = np.uint8(255 * heatmap)
        #heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        #superimposed_img = heatmap * .4 + newImage
        #cv2.imwrite('./data/heatmap.jpg', heatmap)
        #cv2.imwrite('./data/combined.jpg', superimposed_img)
        #exit()
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

        class_correct = list(0. for i in range(2))
        class_total = list(0. for i in range(2))
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


        tn, fp, fn, tp = confusion_matrix(torch.cat(list_of_targets, dim=0).cpu(), torch.cat(list_of_predictions, dim=0).cpu()).ravel()
        print(tn, fp, fn, tp)

        ppv = tp / (tp + fp)
        print('Positive Predictive Value:', ppv)

        npv = tn / (tn + fn)
        print('Negative Predictive Value', npv)

        specificity = tn / (tn + fp)
        print('Specificity:', specificity)

        sensitivity = tp / (tp + fn)
        print('Sensitivity:', sensitivity)

        fpr, tpr, thresholds = roc_curve(torch.cat(list_of_targets, dim=0).cpu(), torch.cat(list_of_probabilities, dim=0).cpu())
        plot_roc_curve(fpr, tpr)
        auc_score = roc_auc_score(torch.cat(list_of_targets, dim=0).cpu(), torch.cat(list_of_probabilities, dim=0).cpu())
        print('AUC Score:', auc_score)
        for i in range(2):
            print('Accuracy of %5s : %2d %%' % (
                classes[i], 100 * class_correct[i] / class_total[i]))

        end_time = datetime.now()
        print('Completed at:', end_time.strftime('%Y-%m-%d %H:%M:%S'))
        duration = end_time - start_time
        print('Elapsed time', duration.total_seconds(), ' seconds')
