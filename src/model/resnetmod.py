'''                                        
Copyright 2024 Image Processing Research Group of University Federico
II of Naples ('GRIP-UNINA'). All rights reserved.
                        
Licensed under the Apache License, Version 2.0 (the "License");       
you may not use this file except in compliance with the License. 
You may obtain a copy of the License at                    
                                           
    http://www.apache.org/licenses/LICENSE-2.0
                                                      
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,    
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.                         
See the License for the specific language governing permissions and
limitations under the License.
'''

import torch
import os
import yaml
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

__all__ = ["ResNet", "resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]


model_urls = {
    "resnet18": "https://download.pytorch.org/models/resnet18-5c106cde.pth",
    "resnet34": "https://download.pytorch.org/models/resnet34-333f7ec4.pth",
    "resnet50": "https://download.pytorch.org/models/resnet50-19c8e357.pth",
    "resnet101": "https://download.pytorch.org/models/resnet101-5d3b4d8f.pth",
    "resnet152": "https://download.pytorch.org/models/resnet152-b121ed2d.pth",
}


class CorviResnetLoader():

    def __init__(self, device, weights_dir='./weights'):
        self.weights_dir = weights_dir
        self.device = device

    def load_model(self):
        model_name = 'Corvi2023'
        _, model_path, arch, norm_type, patch_size = self.get_config(model_name, weights_dir=self.weights_dir)
        model = self.load_weights(self.create_architecture(arch), model_path)
        model = model.to(self.device).eval()
        return model, norm_type, patch_size

    def get_config(self, model_name, weights_dir='./weights'):
        with open(os.path.join(weights_dir, model_name, 'config.yaml')) as fid:
            data = yaml.load(fid, Loader=yaml.FullLoader)
        model_path = os.path.join(weights_dir, model_name, data['weights_file'])
        return data['model_name'], model_path, data['arch'], data['norm_type'], data['patch_size']
    
    def load_weights(self, model, model_path):
        from torch import load
        dat = load(model_path, map_location='cpu')
        if 'model' in dat:
            if ('module._conv_stem.weight' in dat['model']) or \
            ('module.fc.fc1.weight' in dat['model']) or \
            ('module.fc.weight' in dat['model']):
                model.load_state_dict(
                    {key[7:]: dat['model'][key] for key in dat['model']})
            else:
                model.load_state_dict(dat['model'])
        elif 'state_dict' in dat:
            model.load_state_dict(dat['state_dict'])
        elif 'net' in dat:
            model.load_state_dict(dat['net'])
        elif 'main.0.weight' in dat:
            model.load_state_dict(dat)
        elif '_fc.weight' in dat:
            model.load_state_dict(dat)
        elif 'conv1.weight' in dat:
            model.load_state_dict(dat)
        else:
            print(list(dat.keys()))
            assert False
        return model

    def create_architecture(self, name_arch, pretrained=False, num_classes=1):
        if name_arch == "res50nodown":

            if pretrained:
                model = resnet50(pretrained=True, stride0=1, dropout=0.5).change_output(num_classes)
            else:
                model = resnet50(num_classes=num_classes, stride0=1, dropout=0.5)
        elif name_arch == "res50":

            if pretrained:
                model = resnet50(pretrained=True, stride0=2).change_output(num_classes)
            else:
                model = resnet50(num_classes=num_classes, stride0=2)
        else:
            assert False
        return model

class ChannelLinear(nn.Linear):
    def __init__(
        self, in_features: int, out_features: int, bias: bool = True, pool=None
    ) -> None:
        super(ChannelLinear, self).__init__(in_features, out_features, bias)
        self.compute_axis = 1
        self.pool = pool

    def forward(self, x):
        axis_ref = len(x.shape) - 1
        x = torch.transpose(x, self.compute_axis, axis_ref)
        out_shape = list(x.shape)
        out_shape[-1] = self.out_features
        x = x.reshape(-1, x.shape[-1])
        x = x.matmul(self.weight.t())
        if self.bias is not None:
            x = x + self.bias[None, :]
        x = torch.transpose(x.view(out_shape), axis_ref, self.compute_axis)
        if self.pool is not None:
            x = self.pool(x)
        return x


def conv3x3(in_planes, out_planes, stride=1, padding=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=padding, bias=False
    )


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, padding=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride, padding=padding)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, padding=padding)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.padding == 0:
            identity = identity[..., 1:-1, 1:-1]
        if self.downsample is not None:
            identity = self.downsample(identity)
        if self.padding == 0:
            identity = identity[..., 1:-1, 1:-1]

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, padding=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride, padding=padding)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.padding == 0:
            identity = identity[..., 1:-1, 1:-1]
        if self.downsample is not None:
            identity = self.downsample(identity)

        out += identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(
        self,
        block,
        layers,
        num_classes=1000,
        zero_init_residual=False,
        stride0=2,
        padding=1,
        dropout=0.0,
        gap_size=None,
    ):
        super(ResNet, self).__init__()
        self.inplanes = 64

        self.conv1 = nn.Conv2d(
            3, 64, kernel_size=7, stride=stride0, padding=3 * padding, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=stride0, padding=padding)
        self.layer1 = self._make_layer(block, 64, layers[0], padding=padding)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, padding=padding)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, padding=padding)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, padding=padding)

        if gap_size is None:
            self.gap_size = None
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        elif gap_size < 0:
            with torch.no_grad():
                y = self.forward_features(
                    torch.zeros((1, 3, -gap_size, -gap_size), dtype=torch.float32)
                ).shape
            print("gap_size:", -gap_size, ">>", y[-1])
            self.gap_size = y[-1]
            self.avgpool = nn.AvgPool2d(kernel_size=self.gap_size, stride=1, padding=0)
        elif gap_size == 1:
            self.gap_size = gap_size
            self.avgpool = None
        else:
            self.gap_size = gap_size
            self.avgpool = nn.AvgPool2d(kernel_size=self.gap_size, stride=1, padding=0)
        self.num_features = 512 * block.expansion
        self.fc = ChannelLinear(self.num_features, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, padding=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride=stride,
                downsample=downsample,
                padding=padding,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, padding=padding))

        return nn.Sequential(*layers)

    def change_output(self, num_classes):
        self.fc = ChannelLinear(self.num_features, num_classes)
        torch.nn.init.normal_(self.fc.weight.data, 0.0, 0.02)
        return self

    def change_input(self, num_inputs):
        data = self.conv1.weight.data
        old_num_inputs = int(data.shape[1])
        if num_inputs > old_num_inputs:
            times = num_inputs // old_num_inputs
            if (times * old_num_inputs) < num_inputs:
                times = times + 1
            data = data.repeat(1, times, 1, 1) / times
        elif num_inputs == old_num_inputs:
            return self

        data = data[:, :num_inputs, :, :]
        print(self.conv1.weight.data.shape, "->", data.shape)
        self.conv1.weight.data = data

        return self

    def forward_features(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def forward_head(self, x):
        if self.avgpool is not None:
            x = self.avgpool(x)
        if self.dropout is not None:
            x = self.dropout(x)
        y = self.fc(x)
        if self.gap_size is None:
            y = torch.squeeze(torch.squeeze(y, -1), -1)
        return y

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x.squeeze(1)


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls["resnet18"]))
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls["resnet34"]))
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls["resnet50"]))
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls["resnet101"]))
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls["resnet152"]))
    return model