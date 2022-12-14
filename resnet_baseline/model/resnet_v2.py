import torch.nn as nn
import torch


class AttentionBalance(nn.Module):

    def __init__(self, max_threshold=0.7, min_threshold=0.1):
        super(AttentionBalance, self).__init__()
        self.alphaGap = nn.AdaptiveAvgPool2d((1, 1))
        self.alphaGmp = nn.AdaptiveMaxPool2d((1, 1))
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
        self.upsampling = nn.UpsamplingBilinear2d(scale_factor=2)
        self.max_threshold = max_threshold
        self.min_threshold = min_threshold

    def forward(self, z):
        print("am输入：", z.size())
        # torch.Size([64, 512, 28, 28])
        # 64：batch-size
        # 512: 通道数
        avg = self.alphaGap(z)
        max = self.alphaGmp(z)
        print("经过平均池化：", avg.size())
        # torch.Size[64, 512, 1, 1]
        # avg的作用：H，W求平均，得到C*1*1

        avg = self.softmax(avg)
        print("经过第一个softmax：", avg.size())
        # torch.Size([64, 512, 1, 1])
        max = self.softmax(max)
        max = torch.mul(z, max)
        print("与am输入相乘：", max.size())
        # torch.Size([64, 512, 28, 28])
        # torch.Size([64, 512, 28, 28])乘以torch.Size([64, 512, 1, 1])

        max = torch.sum(max, dim=1).unsqueeze(1)
        print("沿着行取最大值：", max.size())
        # torch.Size([64, 1, 28, 28])
        #从torch.Size([64, 512, 28, 28])===>torch.Size([64, 1, 28, 28])

        max = self.upsampling(max)
        print("上采样：", max.size())
        # torch.Size([64, 1, 56, 56])

        max = self.sigmoid(max)

        c = max * self.min_threshold

        max = torch.where(max > self.max_threshold, c, max)
        print("am输出：", max.size())
        # torch.Size([64, 1, 56, 56])

        return avg, max


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=1, stride=1, bias=False)  # squeeze channels
        self.bn1 = nn.BatchNorm2d(out_channel)
        # -----------------------------------------
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channel)
        # -----------------------------------------
        self.conv3 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel*self.expansion,
                               kernel_size=1, stride=1, bias=False)  # unsqueeze channels
        self.bn3 = nn.BatchNorm2d(out_channel*self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, blocks_num, num_classes=1000, include_top=True):
        super(ResNet, self).__init__()
        self.include_top = include_top
        self.in_channel = 64

        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.mask = AttentionBalance(max_threshold=0.8, min_threshold=0.2)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)
        self.alpha_a = nn.AdaptiveAvgPool2d((1, 1))
        self.alpha_m = nn.AdaptiveMaxPool2d((1, 1))
        self.softmax = nn.Softmax(dim=1)
        self.upsampling = nn.UpsamplingBilinear2d(scale_factor=2)
        # self.adaptor = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=1, stride=1, bias=False)
        self.sigmoid = nn.Sigmoid()
        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))

        layers = []
        layers.append(block(self.in_channel, channel, downsample=downsample, stride=stride))
        self.in_channel = channel * block.expansion

        for _ in range(1, block_num):
            layers.append(block(self.in_channel, channel))

        return nn.Sequential(*layers)

    def forward(self, x, attention=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        z = self.layer2(x)
        print("第二层的第一次输入：",x.size())
        # torch.Size([64, 256, 56, 56])
        print("第二层的第一次输出：",z.size())
        # torch.Size([64, 512, 28, 28])
        with torch.no_grad():
            _, mask = self.mask(z)
        #
        print("mul的两个参数：（1）：",x.size(),"（2）：",mask.size())
        # （1）： torch.Size([64, 256, 56, 56]) （2）： torch.Size([64, 1, 56, 56])
        y = torch.mul(x, mask)
        print("相乘后的结果：第二层的第二次输入：",y.size())
        # torch.Size([64, 256, 56, 56])
        x = self.layer2(y)
        x = x + z
        # 注意力分散

        z1 = self.layer3(x)
        with torch.no_grad():

            y1 = self.alpha_m(z1)

            y1 = self.softmax(y1)
            y1 = y1 * z1
            y1 = torch.sum(y1, dim=1).unsqueeze(1)
            y1 = self.upsampling(y1)
            y1 = self.sigmoid(y1)

            y1 = torch.where(y1 > 0.8, y1 * 0.1, y1)
        y1 = torch.mul(x, y1)

        x = self.layer3(y1)

        x = x + z1

        # x = torch.mul(x, avg)
        x = self.layer4(x)

        if self.include_top:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)

        return x


def resnet34(num_classes=1000, include_top=True):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)


def resnet50_v2(num_classes=1000, include_top=True):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)


def resnet101(num_classes=1000, include_top=True):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, include_top=include_top)
