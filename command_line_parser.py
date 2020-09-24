import argparse


def check_positive_integer(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError('Only positive integers are allowed.')
    return value


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a new VGG model or use an existing one on the CIFAR-10 data set.")
    parser.add_argument('-T', '--train FILE',
                        help='Train a new model and save to file (if file exists it will be used to continue training)',
                        dest='train')
    parser.add_argument('-E', '--epochs X', help='Train the model using X epochs (default: 3)',
                        dest='epochs', type=check_positive_integer)
    parser.add_argument('-X', '--execute FILE', help='Execute an existing .pth file on CIFAR-10 data set.',
                        dest='execute')
    parser.add_argument('-N', '--vgg-type TYPE', help='VGG type.  Valid values are VGG11 and VGG16.', dest='vgg_type')
    parser.add_argument('-C', '--cpu', help='Force to run only on CPU.', action='store_true')
    parser.add_argument('-B', '--batch-size', help='Batch size used for training.  (default: 4)', dest='batch_size')
    parser.add_argument('-S', '--competition-size X',
                        help='Train X models and save only the best performing one (least loss)',
                        dest='competition_size',
                        type=check_positive_integer)
    parser.add_argument('-L', '--learn-rate X', help='Learn Rate (default: 0.001)', dest='learn_rate')

    return parser.parse_args()
