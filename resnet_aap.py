import os
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    inplanes = 64

    def __init__(self, block, num_blocks, num_classes=100, in_shape=(3, 224, 224), temperature=1.0):
        super().__init__()
        in_channels, height, width = in_shape

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1  = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64,  num_blocks[0])
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        # Elleniswake
        self.dropout = nn.Dropout(p=0.5)
        self.temperature = temperature

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # Ellenisawake: added Dropout
        x = self.dropout(x)
        x = F.adaptive_avg_pool2d(x, output_size=1)
        x = x.view(x.size(0), -1)
        # Ellenisawake: added Dropout
        x = self.dropout(x)
        x = self.fc(x)
        x = x / self.temperature
        return x


def load_resnet_imagenet_pre_trained(net, pretrained):
    pretrained_dict = torch.load(pretrained)
    model_dict = net.state_dict()
    skip = ['fc.weight', 'fc.bias']
    act_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and k not in skip}
    model_dict.update(act_dict)
    net.load_state_dict(model_dict)


def resnet18(**kwargs):
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    load_resnet_imagenet_pre_trained(model, MODEL_DIR + '/resnet18-5c106cde.pth')
    return model


def resnet34(**kwargs):
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    load_resnet_imagenet_pre_trained(model, MODEL_DIR + '/resnet34-333f7ec4.pth')
    return model


def resnet50(**kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    load_resnet_imagenet_pre_trained(model, MODEL_DIR + '/resnet50-19c8e357.pth')
    return model


def resnet101(**kwargs):
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    load_resnet_imagenet_pre_trained(model, MODEL_DIR + '/resnet101-5d3b4d8f.pth')
    return model


def resnet152(**kwargs):
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    load_resnet_imagenet_pre_trained(model, MODEL_DIR + '/resnet152-b121ed2d.pth')
    return model


class ResFeatExtractor(nn.Module):
    inplanes = 64

    def __init__(self, block, num_blocks, in_shape=(3, 224, 224)):
        super().__init__()
        in_channels, height, width = in_shape

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1  = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64,  num_blocks[0])
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        # Elleniswake
        self.dropout = nn.Dropout(p=0.5)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.adaptive_avg_pool2d(x, output_size=1)
        x = x.view(x.size(0), -1)
        return x


def resnet18backbone(pre_trained='', **kwargs):
    model = ResFeatExtractor(BasicBlock, [2, 2, 2, 2], **kwargs)
    if len(pre_trained) > 0:
        load_resnet_imagenet_pre_trained(model, pre_trained)
    return model


def resnet34backbone(pre_trained='', **kwargs):
    model = ResFeatExtractor(BasicBlock, [3, 4, 6, 3], **kwargs)
    if len(pre_trained) > 0:
        load_resnet_imagenet_pre_trained(model, pre_trained)
    return model


def resnet50backbone(pre_trained='', **kwargs):
    model = ResFeatExtractor(Bottleneck, [3, 4, 6, 3], **kwargs)
    if len(pre_trained) > 0:
        load_resnet_imagenet_pre_trained(model, pre_trained)
    return model


def resnet101backbone(pre_trained='', **kwargs):
    model = ResFeatExtractor(Bottleneck, [3, 4, 23, 3], **kwargs)
    if len(pre_trained) > 0:
        load_resnet_imagenet_pre_trained(model, pre_trained)
    return model


def resnet152backbone(pre_trained='', **kwargs):
    model = ResFeatExtractor(Bottleneck, [3, 8, 36, 3], **kwargs)
    if len(pre_trained) > 0:
        load_resnet_imagenet_pre_trained(model, pre_trained)
    return model


if __name__ == '__main__':
    print('%s: calling main function ... ' % os.path.basename(__file__))