'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

def cosine_distance_func(feat1, feat2):
    # feat1: N * Dim
    # feat2: M * Dim
    # out:   N * M Cosine Distance
    distance = torch.matmul(F.normalize(feat1), F.normalize(feat2).t())
    return distance

def normed_euclidean_distance_func(feat1, feat2):
    # Normalized Euclidean Distance
    # feat1: N * Dim
    # feat2: M * Dim
    # out:   N * M Euclidean Distance
    feat1, feat2 = F.normalize(feat1), F.normalize(feat2)
    feat_matmul  = torch.matmul(feat1, feat2.t())
    distance = torch.ones_like(feat_matmul) - feat_matmul
    distance = distance * 2
    return distance.clamp(1e-10).sqrt()

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, clsr='linear', num_proto=1):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.clsr = clsr
        self.num_proto = num_proto
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        if self.clsr == 'linear':
            self.linear = nn.Linear(512*block.expansion, num_classes)
        elif self.clsr == 'nce':
            self.conv5  = nn.Conv2d(512, 512, kernel_size=1, bias=False)
            self.protos = nn.Parameter(
                torch.randn(self.num_proto * self.num_classes, 512),
                requires_grad=True
            )
            self.radius = nn.Parameter(
                torch.rand(1, self.num_proto * self.num_classes) - 0.5,
                requires_grad=True
            )
        else:
            raise ValueError('Classifier {} is not supported!'.format(self.clsr))

    def nce_prob_cos(self, feat):
        dist = cosine_distance_func(feat, self.protos)
        dist = (dist / self.radius.sigmoid()).sigmoid()
        cls_score, _ = dist.view(-1, self.num_proto, self.num_classes).max(1)
        return cls_score

    def nce_prob_euc(self, feat):
        dist = normed_euclidean_distance_func(feat.sigmoid(), self.protos.sigmoid())
        cls_score, _ = dist.view(-1, self.num_proto, self.num_classes).max(1)
        cls_score = torch.exp(-(cls_score ** 2) / (2 * self.radius.sigmoid() ** 2))
        return cls_score

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, is_feat=False, is_feats=False):
        
        out = F.relu(self.bn1(self.conv1(x)))
        out1 = self.layer1(out)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        out = F.avg_pool2d(out4, 4)

        if self.clsr == 'linear':
            feat = out.view(out.size(0), -1)
            out = self.linear(feat)
        elif self.clsr == 'nce':
            feat = self.conv5(out).view(out.size(0), -1)
            out = self.nce_prob_euc(feat)
        

        if is_feat:
            return out, feat
        elif is_feats:
            return out, [out1, out2, out3, out4]
        else:
            return out


def ResNet18(num_classes=10, num_proto=1, clsr='linear', pretrained=False):
    return ResNet(
        BasicBlock,
        [2, 2, 2, 2],
        num_classes=num_classes,
        num_proto=num_proto,
        clsr=clsr,
    )

def ResNet34(num_classes=10):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes,)


def ResNet50(num_classes=10):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes,)


def ResNet101(num_classes=10):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes,)


def ResNet152(num_classes=10):
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes=num_classes,)


def test():
    net = ResNet18()
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())

# test()
