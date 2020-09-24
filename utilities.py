import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from vgg import VggNet


def results_sorter(value):
    return value['loss']


def plot_roc_curve(fpr, tpr, filename='figure.png', color='orange', line_color='darkblue'):
    plt.plot(fpr, tpr, color=color, label='ROC')
    plt.plot([0, 1], [0, 1], color=line_color, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.savefig(filename)
    #plt.show()


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