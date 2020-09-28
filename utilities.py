import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from vgg import VggNet


def results_sorter(value):
    return value['loss']


def plot_roc_curve(fpr, tpr, filename='roc_curve.png', color='orange', line_color='darkblue', prefix=''):
    plt.figure(1)
    plt.plot(fpr, tpr, color=color, label='ROC')
    plt.plot([0, 1], [0, 1], color=line_color, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.savefig(f'./results/{prefix}{filename}')
    plt.close(1)


def plot_epoch_losses(epoch_losses, filename='epoch_losses.png', color='green', line_color='green', prefix=''):
    lowest_loss_epoch = np.argmin(epoch_losses)
    plt.figure(2)
    plt.plot(epoch_losses, color=color)
    plt.axvline(lowest_loss_epoch, 0, 1, label=f'Lowest Loss Epoch: {lowest_loss_epoch}')
    plt.xlabel('Epoch')
    plt.ylabel('Average loss')
    plt.title('Average loss per epoch')
    plt.legend()
    plt.savefig(f'./results/{prefix}{filename}')
    plt.close(2)

def create_train_net(vgg_type_param, device_param, state_dict=None, optimizer_state_dict=None, learn_rate=0.001,
                     num_classes=10, size=32):
    train_net = VggNet(in_channels=3, num_classes=num_classes, size=size,
                       vgg_type=vgg_type_param, device=device_param).to(device_param)
    if state_dict is not None:
        train_net.load_state_dict(state_dict)
    if torch.cuda.device_count() > 1:
        print('Attempting to use', torch.cuda.device_count(), 'GPUs')
        train_net = nn.DataParallel(train_net).to(device_param)
    train_net.train()
    train_optimizer = optim.SGD(train_net.parameters(), lr=learn_rate, momentum=0.9)
    if optimizer_state_dict is not None:
        train_optimizer.load_state_dict(optimizer_state_dict)
    return train_net, train_optimizer


def print_model_metrics(targets, predictions, probabilities, threshold, prefix=''):
    with open(f'./results/{prefix}metrics.txt', 'w') as f:
        tn, fp, fn, tp = confusion_matrix(torch.cat(targets, dim=0).cpu(),
                                          torch.cat(predictions, dim=0).cpu()).ravel()
        print(tn, fp, fn, tp)

        ppv = tp / (tp + fp)
        print('Positive Predictive Value:', ppv)
        print('Positive Predictive Value:', ppv, file=f)

        npv = tn / (tn + fn)
        print('Negative Predictive Value', npv)
        print('Negative Predictive Value', npv, file=f)

        specificity = tn / (tn + fp)
        print('Specificity:', specificity)
        print('Specificity:', specificity, file=f)

        sensitivity = tp / (tp + fn)
        print('Sensitivity:', sensitivity)
        print('Sensitivity:', sensitivity, file=f)

        fpr, tpr, thresholds = roc_curve(torch.cat(targets, dim=0).cpu(),
                                         torch.cat(probabilities, dim=0).cpu())
        plot_roc_curve(fpr, tpr, prefix=prefix)
        auc_score = roc_auc_score(torch.cat(targets, dim=0).cpu(),
                                  torch.cat(probabilities, dim=0).cpu())
        print('AUC Score:', auc_score)
        print('AUC Score:', auc_score, file=f)

        print('Threshold:', threshold)
        print('Threshold:', threshold, file=f)


def print_execution_summary(classes, class_correct, class_total, start_time, end_time):
    for i in range(len(classes)):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))

    print('Completed at:', end_time.strftime('%Y-%m-%d %H:%M:%S'))
    duration = end_time - start_time
    print('Elapsed time', duration.total_seconds(), ' seconds')


def print_training_summary(best_accuracy, start_time, end_time, epoch_losses, prefix=''):
    print('Network with accuracy of ', best_accuracy, 'on validation data set chosen.')
    print('Finished Training')
    print('Completed at:', end_time.strftime('%Y-%m-%d %H:%M:%S'))
    duration = end_time - start_time
    print('Elapsed time', duration.total_seconds(), ' seconds')
    plot_epoch_losses(epoch_losses, prefix=prefix)


def save_model(model, filename):
    torch.save(model, filename)
    print('File saved to ', filename)
