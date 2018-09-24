import torch

import torch.nn as nn


def convolution_block(filters, size, stride=1, padding=1, activation=True):
    layers = [
        nn.Conv2d(filters, size, strides=stride, padding=padding),
        nn.BatchNorm2d(),
    ]
    if activation:
        layers.append(nn.ReLU())

    return nn.Sequential(*layers)


def residual_block(num_filters=16):
    layers = [nn.ReLU(),
              nn.BatchNorm2d(),
              convolution_block(num_filters, 3),
              convolution_block(num_filters, 3, activation=False)]
    # x = Add()([x, blockInput])
    return nn.Sequential(*layers)


class Unet(nn.Module):
    def __init__(self, start_neurons, dropout_ratio=0.5):
        conv1 = nn.Conv2d(start_neurons * 1, (3, 3), padding=0)
        conv1 = residual_block(conv1, start_neurons * 1)
        conv1 = residual_block(conv1, start_neurons * 1)
        conv1 = nn.ReLU()
        pool1 = nn.MaxPool2d((2, 2))
        pool1 = nn.Dropout2d(dropout_ratio / 2)

        # 50 -> 25
        conv2 = nn.Conv2d(start_neurons * 2, (3, 3),  padding=0)
        conv2 = residual_block(conv2, start_neurons * 2)
        conv2 = residual_block(conv2, start_neurons * 2)
        conv2 = nn.ReLU()
        pool2 = nn.MaxPool2d((2, 2))
        pool2 = nn.Dropout2d(dropout_ratio)

        # 25 -> 12
        conv3 = nn.Conv2d(start_neurons * 4, (3, 3), padding=0)
        conv3 = residual_block(conv3, start_neurons * 4)
        conv3 = residual_block(conv3, start_neurons * 4)
        conv3 = nn.ReLU()
        pool3 = nn.MaxPool2d((2, 2))
        pool3 = nn.Dropout2d(dropout_ratio)

        # 12 -> 6
        conv4 = nn.Conv2d(start_neurons * 8, (3, 3), padding=0)
        conv4 = residual_block(conv4, start_neurons * 8)
        conv4 = residual_block(conv4, start_neurons * 8)
        conv4 = nn.ReLU()
        pool4 = nn.MaxPool2d((2, 2))
        pool4 = nn.Dropout2d(dropout_ratio)

        # Middle
        convm = nn.Conv2d(start_neurons * 16, (3, 3), padding=0)
        convm = residual_block(convm, start_neurons * 16)
        convm = residual_block(convm, start_neurons * 16)
        convm = nn.ReLU()(convm)

        # 6 -> 12
        deconv4 = nn.ConvTranspose2d(start_neurons * 8, (3, 3), strides=(2, 2), padding="same")
        uconv4 = torch.cat([deconv4, conv4])
        uconv4 = nn.Dropout2d(dropout_ratio)

        uconv4 = nn.Conv2d(start_neurons * 8, (3, 3), activation=None, padding="same")(uconv4)
        uconv4 = residual_block(uconv4, start_neurons * 8)
        uconv4 = residual_block(uconv4, start_neurons * 8)
        uconv4 = nn.ReLU()(uconv4)

        # 12 -> 25
        # deconv3 = nn.Conv2dTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="same")(uconv4)
        deconv3 = nn.Conv2dTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="valid")(uconv4)
        uconv3 = torch.cat([deconv3, conv3])
        uconv3 = nn.Dropout2d(dropout_ratio)(uconv3)

        uconv3 = nn.Conv2d(start_neurons * 4, (3, 3), activation=None, padding="same")(uconv3)
        uconv3 = residual_block(uconv3, start_neurons * 4)
        uconv3 = residual_block(uconv3, start_neurons * 4)
        uconv3 = nn.ReLU()(uconv3)

        # 25 -> 50
        deconv2 = nn.Conv2dTranspose(start_neurons * 2, (3, 3), strides=(2, 2), padding="same")(uconv3)
        uconv2 = torch.cat([deconv2, conv2])

        uconv2 = nn.Dropout2d(dropout_ratio)(uconv2)
        uconv2 = nn.Conv2d(start_neurons * 2, (3, 3), activation=None)
        uconv2 = residual_block(uconv2, start_neurons * 2)
        uconv2 = residual_block(uconv2, start_neurons * 2)
        uconv2 = nn.ReLU()(uconv2)

        # 50 -> 101
        # deconv1 = nn.Conv2dTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="same")(uconv2)
        deconv1 = nn.Conv2dTranspose(start_neurons * 1, (3, 3), strides=(2, 2), 0)
        uconv1 = torch.cat([deconv1, conv1])

        uconv1 = nn.Dropout2d(dropout_ratio)
        uconv1 = nn.Conv2d(start_neurons * 1, (3, 3), padding=0)
        uconv1 = residual_block(uconv1, start_neurons * 1)
        uconv1 = residual_block(uconv1, start_neurons * 1)
        uconv1 = nn.ReLU()

        uconv1 = nn.Dropout2d(dropout_ratio / 2)
        output_layer = nn.Conv2d(1, (1, 1), padding=0)
