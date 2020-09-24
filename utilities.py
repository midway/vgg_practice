import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from vgg import VggNet


def results_sorter(value):
    return value['loss']


def plot_roc_curve(fpr, tpr, filename='figure.png', color='orange', line_color='darkblue'):
    plt.figure(1)
    plt.plot(fpr, tpr, color=color, label='ROC')
    plt.plot([0, 1], [0, 1], color=line_color, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.savefig(filename)


def plot_epoch_losses(epoch_losses, filename='epoch_losses.png', color='green', line_color='darkblue'):
    plt.figure(2)
    plt.plot(epoch_losses, color=color)
    plt.xlabel('Epoch')
    plt.ylabel('Average loss')
    plt.title('Average loss per epoch')
    plt.legend()
    plt.savefig(filename)


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


def print_model_metrics(targets, predictions, probabilities):
    tn, fp, fn, tp = confusion_matrix(torch.cat(targets, dim=0).cpu(),
                                      torch.cat(predictions, dim=0).cpu()).ravel()
    print(tn, fp, fn, tp)

    ppv = tp / (tp + fp)
    print('Positive Predictive Value:', ppv)

    npv = tn / (tn + fn)
    print('Negative Predictive Value', npv)

    specificity = tn / (tn + fp)
    print('Specificity:', specificity)

    sensitivity = tp / (tp + fn)
    print('Sensitivity:', sensitivity)

    fpr, tpr, thresholds = roc_curve(torch.cat(targets, dim=0).cpu(),
                                     torch.cat(probabilities, dim=0).cpu())
    plot_roc_curve(fpr, tpr)
    auc_score = roc_auc_score(torch.cat(targets, dim=0).cpu(),
                              torch.cat(probabilities, dim=0).cpu())
    print('AUC Score:', auc_score)

