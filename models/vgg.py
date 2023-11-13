'''VGG11/13/16/19 in Pytorch.'''
from fcntl import F_GETLK64
import torch
import torch.nn as nn


cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name, num_classes=10, pretrained=False):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x, is_feat=False, is_feats=False):
        if is_feats:
            x = self.features[:6](x)
            f0 = x
            x = self.features[6:13](x)
            f1 = x
            x = self.features[13:23](x)
            f2 = x
            x = self.features[23:33](x)
            f3 = x
            x = self.features[33:43](x)
            f4 = x
            x = self.features[43:](x)
            x = x.view(x.size(0), -1)
            f5 = x
            out = self.classifier(x)
            return out, [f0, f1, f2, f3, f4, f5]
        else:
            out = self.features(x)
            feat = out.view(out.size(0), -1)
            out = self.classifier(feat)
            if is_feat:
                return out, feat
            else:
                return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

def vgg11_bn(num_classes=10, pretrained=False):
    return VGG('VGG11', num_classes=num_classes, pretrained=pretrained)

def vgg13_bn(num_classes=10, pretrained=False):
    return VGG('VGG13', num_classes=num_classes, pretrained=pretrained)

def vgg16_bn(num_classes=10, pretrained=False):
    return VGG('VGG16', num_classes=num_classes, pretrained=pretrained)

def vgg19_bn(num_classes=10, pretrained=False):
    return VGG('VGG19', num_classes=num_classes, pretrained=pretrained)

def test():
    net = VGG('VGG11')
    x = torch.randn(2,3,32,32)
    y = net(x)
    print(y.size())

# test()
