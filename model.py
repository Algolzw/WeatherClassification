import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models
import pretrainedmodels as pmodels
from efficientnet_pytorch import EfficientNet
from multigrain.lib import get_multigrain
from fixres.pnasnet import pnasnet5large


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/ig_resnext101_32x8-c38310e5.pth',
    'resnext101_32x16d': 'https://download.pytorch.org/models/ig_resnext101_32x16-c6f796b0.pth',
    'resnext101_32x32d': 'https://download.pytorch.org/models/ig_resnext101_32x32-e4b90b00.pth',
    'resnext101_32x48d': 'https://download.pytorch.org/models/ig_resnext101_32x48-3e41cc8a.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
    'densenet121':'https://download.pytorch.org/models/densenet121-a639ec97.pth',
    'densenet201': 'https://download.pytorch.org/models/densenet201-c1103571.pth',
    'densenet161': 'https://download.pytorch.org/models/densenet161-8d451a50.pth',
    'inception_v3': 'https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth',
    'fixpnas': '/home/lzw/.cache/torch/checkpoints/PNASNet.pth'
}

def resnet(num_classes=9, layers=101, state_dict=None):
    if layers == 18:
        model = models.resnet18()
    elif layers == 34:
        model = models.resnet34()
    elif layers == 50:
        model = models.resnet50()
    elif layers == 101:
        model = models.resnet101()
    elif layers == 152:
        model = models.resnet152()

    if state_dict is not None:
        print('load_state_dict')
        model.load_state_dict(state_dict)

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    return model

def resnext(num_classes=9, layers=101, state_dict=None):
    if layers == 50:
        model = models.resnext50_32x4d()
    elif layers == 101:
        model = models.resnext101_32x8d()

    if state_dict is not None:
        model.load_state_dict(state_dict)

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    return model

def resnext_wsl(num_classes=9, bottleneck_width=8):
    model = torch.hub.load('facebookresearch/WSL-Images', 'resnext101_32x'+str(bottleneck_width)+'d_wsl')

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model

def resnext_swsl(num_classes=9, layers=101, bottleneck_width=8):
    model = torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models', 'resnext'+str(layers)+'_32x'+str(bottleneck_width)+'d_swsl')

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model

def vgg_bn(num_classes=9, layers=16, state_dict=None):
    if layers == 16:
        model = models.vgg16_bn()
    elif layers == 19:
        model = models.vgg19_bn()

    if state_dict is not None:
        model.load_state_dict(state_dict)

    model._modules['6'] = nn.Linear(4096, num_classes)
    return model

def densenet(num_classes=9, layers=121, state_dict=None):
    '''
        layers: 121, 201, 161
    '''
    if layers == 121:
        model = models.densenet121()
    elif layers == 201:
        model = models.densenet201()
    elif layers == 161:
        model = models.densenet161()

    if state_dict is not None:
        model.load_state_dict(state_dict)

    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, num_classes)
    return model

def inception_v3(num_classes=9, layers=101, state_dict=None):
    model = models.inception_v3()
    if state_dict is not None:
        model.load_state_dict(state_dict)

    aux_ftrs = model.AuxLogits.fc.in_features
    model.AuxLogits.fc = nn.Linear(aux_ftrs, num_classes)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model

def dpn(num_classes=9, layers=92, pretrained=True):
    model = torch.hub.load('rwightman/pytorch-dpn-pretrained', 'dpn'+str(layers), pretrained=pretrained)

    in_chs = model.classifier.in_channels
    model.classifier = nn.Conv2d(in_chs, num_classes, kernel_size=1, bias=True)
    return model

class EffNet(nn.Module):
    def __init__(self, num_classes=9, layers=0, pretrained=False):
        super(EffNet, self).__init__()
        if pretrained:
            self.model = EfficientNet.from_pretrained('efficientnet-b'+str(layers))
        else:
            self.model = EfficientNet.from_name('efficientnet-b'+str(layers))

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))

        num_ftrs = self.model._fc.in_features
        self._fc = nn.Sequential(
            nn.BatchNorm1d(num_ftrs*2),
            nn.Dropout(inplace=True),
            nn.Linear(num_ftrs*2, num_ftrs, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(num_ftrs),
            nn.Dropout(inplace=True),
            nn.Linear(num_ftrs, num_classes, bias=False)
            )

    def forward(self, x):
        x = self.model.extract_features(x)
        avgfeature = torch.flatten(self.avgpool(x), 1)
        maxfeature = torch.flatten(self.maxpool(x), 1)
        x = torch.cat([avgfeature, maxfeature], 1)
        return self._fc(x)

    def extract_features(self, x):
        x = self.model.extract_features(x)
        avgfeature = torch.flatten(self.avgpool(x), 1)
        maxfeature = torch.flatten(self.maxpool(x), 1)
        x = torch.cat([avgfeature, maxfeature], 1)
        return x

def effnet(num_classes=9, layers=0, pretrained=False):
    return EffNet(num_classes, layers, pretrained)

# def effnet(num_classes=9, layers=0, pretrained=False):
#     if pretrained:
#         model = EfficientNet.from_pretrained('efficientnet-b'+str(layers))
#     else:
#         model = EfficientNet.from_name('efficientnet-b'+str(layers))
#     num_ftrs = model._fc.in_features

#     model._fc = nn.Linear(num_ftrs, num_classes)
#     return model

def pnasnet_m(num_classes=9, layers=5, pretrained=False):
    model = get_multigrain(backbone='pnasnet5large', include_sampling=False, learn_p=False, p=1.7)
    if pretrained:
        model.load_state_dict(
            torch.load('/home/lzw/.cache/torch/checkpoints/pnasnet5large-finetune500.pth')['model_state'])
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, num_classes)
    return model

def senet_m(num_classes=9, layers=154, pretrained=False):
    model = get_multigrain(backbone='senet154', include_sampling=False, learn_p=False, p=1.6)
    if pretrained:
        model.load_state_dict(
            torch.load('/home/lzw/.cache/torch/checkpoints/senet154-finetune400.pth')['model_state'])
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, num_classes)
    return model

def cadene_model(num_classes=9, model_name='inceptionresnetv2'):
    model = pmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')

    if model_name == 'inceptionresnetv2':
        model.avgpool_1a = nn.AdaptiveAvgPool2d((1, 1))
    elif model_name[:5] in ['resne', 'senet', 'pnasn', 'nasne', 'polyn', 'se_re']:
        model.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

    num_ftrs = model.last_linear.in_features
    model.last_linear = nn.Linear(num_ftrs, num_classes)
    return model

def fixpnas(num_classes=9, pretrained=False):
    model = pnasnet5large(pretrained=None)
    if pretrained:
        pretrained_dict=torch.load(model_urls['fixpnas'],map_location='cpu')['model']
        model_dict = model.state_dict()
        for k in model_dict.keys():
            if(('module.'+k) in pretrained_dict.keys()):
                model_dict[k]=pretrained_dict.get(('module.'+k))
        model.load_state_dict(model_dict)
    else:
        model = pnasnet5large(pretrained=None)
    num_ftrs = model.last_linear.in_features
    model.last_linear = nn.Linear(num_ftrs, num_classes)
    return model














