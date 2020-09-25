# Implementation of VGG16 as described in https://arxiv.org/pdf/1409.1556
# "VERY DEEP CONVOLUTIONAL NETWORKS FOR LARGE-SCALE IMAGE RECOGNITION"
#
# also based on the code-along tutorial at https://youtu.be/ACmuBbuXn20 but
# with additional sourcing back to the original paper and explanations to
# myself for reference
import torch
import torch.nn as nn

# Convolutional layers
# Integer value means outputs, M is for maxpool
VGG = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
}

# constants as defined in the paper
# p.2 2.1
conv_kernel_size = (3, 3)
conv_stride = (1, 1)
conv_padding = (1, 1)
max_pool_kernel_size = (2, 2)
max_pool_stride = (2, 2)


class VggNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=1000, size=224, vgg_type='VGG16', device='cuda'):
        super(VggNet, self).__init__()
        self.device = device
        self.in_channels = in_channels
        self.convolution_layers = self.create_conv_layers(VGG[vgg_type])

        self.max_pool = nn.MaxPool2d(kernel_size=max_pool_kernel_size, stride=max_pool_stride)
        self.gradients = None

        self.fully_connected_layers = nn.Sequential(

            # 512 because 512 is the number of outputs of the last
            # convolutional layer defined for VGG16
            # 7 because we have 5 maxpools so 224 / (2 ^ 5) = 7
            # 4096 because that is what is stated in the paper
            # p.3 Table 1
            nn.Linear(int(512 * (size / 32)**2), 4096),

            # p.2 2.1  (ReLU is short for rectified linear unit, an activation function)
            nn.ReLU(),

            # p.4 3.1 (Dropout is for reducing overfit data)
            nn.Dropout(p=0.5),

            # second Linear layer with 4096 input to match 4096 outputs of first linear layer
            nn.Linear(4096, 4096),

            # p.2 2.1
            nn.ReLU(),

            # p.4 3.1
            nn.Dropout(p=0.5),

            # third Linear layer with 4096 input to match 4096 outputs of first linear layer
            # and the number of outputs to equal the number of classes we have
            nn.Linear(4096, num_classes),
        )

    def activations_hook(self, grad):
        self.gradients = grad

    def forward(self, x):
        # call the convolution layers
        # "shape after convolution layers should be 1 x 3 x 244 x 244
        x = self.convolution_layers(x.to(self.device))

        # register the hook
        h = x.register_hook(self.activations_hook)

        x = self.max_pool(x)

        # reshape the convolutional layers into a linear structure (1 dimension) for
        # processing by the nn.Sequential code
        x = x.reshape(x.shape[0], -1)
        # shape after reshape should be 1 x 25088
        # call our fully connected layer code with the newly reshaped x
        x = self.fully_connected_layers(x)
        return x

    # method for the gradient extraction
    def get_activations_gradient(self):
        return self.gradients

    # method for the activation exctraction
    def get_activations(self, x):
        return self.convolution_layers(x.to(self.device))

    def create_conv_layers(self, architectures):
        layers = []
        # start with in_channels as the initial input channel size from
        # the class creation, in this case, 3
        in_channels = self.in_channels

        for x in architectures:
            if type(x) == int:
                out_channels = x
                # create the convolution layers with input and
                # output channel counts matching our current needs
                # along with a stride and padding as defined in the paper
                #
                # ReLU applied to all convolution layers
                layers += [nn.Conv2d(in_channels=in_channels,
                                     out_channels=out_channels,
                                     kernel_size=conv_kernel_size,
                                     stride=conv_stride,
                                     padding=conv_padding),
                           nn.BatchNorm2d(x),
                           nn.ReLU()]

                # the input channels for the next layer need to match the output channel
                # count of the previous layer
                in_channels = x
            elif x == 'M':
                layers += [nn.MaxPool2d(kernel_size=max_pool_kernel_size, stride=max_pool_stride)]
        sequential = nn.Sequential(*(layers[:-1]))
        for _ in sequential.parameters():
            _.requires_grad_(True)
        return sequential.requires_grad_(True)

