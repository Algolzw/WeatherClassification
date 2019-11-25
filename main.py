import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from tensorboardX import SummaryWriter
from torchsummary import summary
import smote_variants as sv

import numpy as np
import random
import os
import re
import copy

import utils
import opts
from train import train_model
from model import *
from dataloader import WeatherDataset, my_transform
from lr_scheduler import WarmupMultiStepLR, WarmupCosineAnnealingLR, CosineAnnealingWithRestartsLR
from label_smooth import LabelSmoothSoftmaxCE, FocalLoss2
from optimizer import RAdam, PlainRAdam, AdamW, NovoGrad, Ranger, Ralamb, Lookahead
from sync_batchnorm import convert_model


# os.environ["CUDA_VISIBLE_DEVICES"] = "4，5，6，7"

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/ig_resnext101_32x8-c38310e5.pth',
    'resnext101_32x16d': 'https://download.pytorch.org/models/ig_resnext101_32x16-c6f796b0.pth',
    'resnext101_32x32d': 'https://download.pytorch.org/models/ig_resnext101_32x32-e4b90b00.pth',
    'resnext101_32x48d': 'https://download.pytorch.org/models/ig_resnext101_32x48-3e41cc8a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
    'densenet121':'https://download.pytorch.org/models/densenet121-a639ec97.pth',
    'densenet201': 'https://download.pytorch.org/models/densenet201-c1103571.pth',
    'densenet161': 'https://download.pytorch.org/models/densenet161-8d451a50.pth',
    'inception_v348': 'https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth',
    'dpn92': 'https://github.com/rwightman/pytorch-dpn-pretrained/releases/download/v0.1/dpn92_extra-b040e4a9b.pth',
    'dpn98': 'https://github.com/rwightman/pytorch-dpn-pretrained/releases/download/v0.1/dpn98-5b90dec4d.pth',
    'dpn131': 'https://github.com/rwightman/pytorch-dpn-pretrained/releases/download/v0.1/dpn131-71dfe43e0.pth',
    'dpn107': 'https://github.com/rwightman/pytorch-dpn-pretrained/releases/download/v0.1/dpn107_extra-1ac7121e2.pth',
    'effnet0': 'http://storage.googleapis.com/public-models/efficientnet/efficientnet-b0-355c32eb.pth',
    'effnet1': 'http://storage.googleapis.com/public-models/efficientnet/efficientnet-b1-f1951068.pth',
    'effnet2': 'http://storage.googleapis.com/public-models/efficientnet/efficientnet-b2-8bb594d6.pth',
    'effnet3': 'http://storage.googleapis.com/public-models/efficientnet/efficientnet-b3-5fb5a3c3.pth',
    'effnet4': 'http://storage.googleapis.com/public-models/efficientnet/efficientnet-b4-6ed6700e.pth',
    'effnet5': 'http://storage.googleapis.com/public-models/efficientnet/efficientnet-b5-b6417697.pth',
    'effnet6': 'http://storage.googleapis.com/public-models/efficientnet/efficientnet-b6-c76e70fd.pth',
    'effnet7': 'http://storage.googleapis.com/public-models/efficientnet/efficientnet-b7-dcc49843.pth',
    'pnasnet_m5': '/home/lzw/.cache/torch/checkpoints/pnasnet5large-finetune500.pth',
}

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True # cpu\gpu 结果一致

def get_optimizer(opt, params, weight_decay=1e-4):
    # large_lr_layers = list(map(id, model.module._fc.parameters()))
    # small_lr_layers = filter(lambda p:id(p) not in large_lr_layers, model.module.parameters())

    if opt.optimizer == 'sgd':
        optimizer = optim.SGD(params, lr=opt.lr, momentum=0.9, nesterov=True, weight_decay=weight_decay)
        # optimizer = torch.optim.SGD([
        #             {"params":model.fc.parameters()}
        #             # {"params":small_lr_layers,"lr":opt.lr/10}
        #             ],lr = opt.lr, momentum=0.9, weight_decay=1e-4)
    elif opt.optimizer == 'adam':
        optimizer = optim.Adam(params, lr=opt.lr, weight_decay=weight_decay)
    elif opt.optimizer == 'radam':
        optimizer = RAdam(params, lr=opt.lr, weight_decay=weight_decay)
    elif opt.optimizer == 'adamw':
        optimizer = optim.AdamW(params, lr=opt.lr, amsgrad=True, weight_decay=weight_decay)
        # optimizer = torch.optim.AdamW([
        #             {"params":model.module._fc.parameters()},
        #             {"params":small_lr_layers,"lr":opt.lr/10}
        #             ],lr = opt.lr, weight_decay=5e-4)
    elif opt.optimizer == 'rms':
        # optimizer = optim.RMSprop([
        #                 {"params":model.module._fc.parameters()},
        #                 {"params":small_lr_layers, "lr": opt.lr/10}
        #                 ], lr=opt.lr, momentum=0.9, weight_decay=1e-4)
        optimizer = optim.RMSprop(params, lr=opt.lr, momentum=0.9, weight_decay=weight_decay)
    elif opt.optimizer == 'novograd':
        optimizer = NovoGrad(params, lr=opt.lr, grad_averaging=True)
    elif opt.optimizer == 'ranger':
        optimizer = Ranger(params, lr=opt.lr, weight_decay=weight_decay)
    elif opt.optimizer == 'ralamb':
        optimizer = Ralamb(params, lr=opt.lr, weight_decay=weight_decay)

    if opt.lookahead:
        optimizer = Lookahead(optimizer, k=5, alpha=0.5)
    return optimizer

def get_schedule(opt, optimizer, train_loader_len=None):
    if opt.scheduler == 'multistep':
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60, 100, 130], gamma=0.1)
    elif opt.scheduler == 'cycle':
        step_size = train_loader_len*4
        print(step_size)
        scheduler = lr_scheduler.CyclicLR(optimizer, step_size_up=step_size, base_lr=opt.lr/100, max_lr=opt.lr, cycle_momentum=False)
    elif opt.scheduler == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
    elif opt.scheduler == 'warmup':
        step = train_loader_len
        scheduler = WarmupMultiStepLR(optimizer, milestones=[step*30, step*60, step*100, step*130], gamma=0.1)
    elif opt.scheduler == 'cos':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, train_loader_len*5, eta_min=1e-8)
    elif opt.scheduler == 'cosw':
        scheduler = WarmupCosineAnnealingLR(optimizer, train_loader_len*5, eta_min=1e-8)
    elif opt.scheduler == 'sgdr':
        scheduler = CosineAnnealingWithRestartsLR(optimizer, train_loader_len*5, eta_min=1e-10, T_mult=1.1)
    elif opt.scheduler == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    elif opt.scheduler == 'exponential':
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.97)
    else:
        scheduler = None

    return scheduler

def get_model(opt):
    state_dict = None
    if opt.pretrained and opt.network+str(opt.layers) in model_urls.keys():
        state_dict = torch.utils.model_zoo.load_url(model_urls[opt.network+str(opt.layers)])

    if opt.cadene:
        model = cadene_model(opt.classes, model_name=opt.network)
    elif opt.network == 'resnet':
        model = resnet(opt.classes, opt.layers, state_dict)
    elif opt.network == 'resnext':
        model = resnext(opt.classes, opt.layers, state_dict)
    elif opt.network == 'resnext_wsl':
        # resnext_wsl must specify the opt.battleneck_width parameter
        opt.network = 'resnext_wsl_32x' + str(opt.battleneck_width) +'d'
        model = resnext_wsl(opt.classes, opt.battleneck_width)
    elif opt.network == 'resnext_swsl':
        model = resnext_swsl(opt.classes, opt.layers, opt.battleneck_width)
    elif opt.network == 'vgg':
        model = vgg_bn(opt.classes, opt.layers, state_dict)
    elif opt.network == 'densenet':
        pattern = re.compile(
            r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
        for key in list(state_dict.keys()):
            res = pattern.match(key)
            if res:
                new_key = res.group(1) + res.group(2)
                state_dict[new_key] = state_dict[key]
                del state_dict[key]
        model = densenet(opt.classes, opt.layers, state_dict)
    elif opt.network == 'inception_v3':
        model = inception_v3(opt.classes, opt.layers, state_dict)
    elif opt.network == 'dpn':
        model = dpn(opt.classes, opt.layers, opt.pretrained)
    elif opt.network == 'effnet':
        model = effnet(opt.classes, opt.layers, opt.pretrained)
    elif opt.network == 'pnasnet_m':
        model = pnasnet_m(opt.classes, opt.layers, opt.pretrained)
    elif opt.network == 'senet_m':
        model = senet_m(opt.classes, opt.layers, opt.pretrained)
    elif opt.network == 'fixpnas':
        model = fixpnas(opt.classes, opt.pretrained)

    return model


def main(opt):
    setup_seed(opt.seed)
    if torch.cuda.is_available():
        device = torch.device('cuda')
        torch.cuda.set_device(opt.gpu_id)
    else:
        device = torch.device('cpu')

    log_dir = opt.log_dir+'/'+opt.network+'-'+str(opt.layers)
    utils.mkdir(log_dir)

    model = get_model(opt)
    # model = nn.DataParallel(model, device_ids=[1, 2, 3])
    # model = nn.DataParallel(model, device_ids=[0, 1, 2, 3])
    # model = nn.DataParallel(model, device_ids=[4, 5, 6, 7])
    model = nn.DataParallel(model, device_ids=[0, 1, 2, 3, 4, 5, 6, 7])
    # model = convert_model(model)
    model = model.to(device)

    summary_writer = SummaryWriter(logdir=log_dir)
    weight = None
    if opt.classes == 9:
        weight = torch.tensor([1.8, 1, 1, 1.2, 1, 1.6, 1.2, 1.4, 1], device=device)
    elif opt.classes == 8:
        weight = torch.tensor([1.8, 1, 1.2, 1.6, 1, 1.2, 1.8, 1], device=device)
    elif opt.classes == 2:
        weight = torch.tensor([1., 1.5], device=device)

    if opt.criterion == 'lsr':
        criterion = LabelSmoothSoftmaxCE(weight=weight, use_focal_loss=opt.use_focal, reduction='sum').cuda()
    elif opt.criterion == 'focal':
        # criterion = FocalLoss(alpha=1, gamma=2, reduction='sum')
        criterion = FocalLoss2()
    elif opt.criterion == 'ce':
        criterion = nn.CrossEntropyLoss(weight=weight, reduction='sum').cuda()
    elif opt.criterion == 'bce':
        criterion = nn.BCEWithLogitsLoss(weight=weight, reduction='sum').cuda()

    if opt.classes > 2:
        # all data
        images, labels = utils.read_data(
                os.path.join(opt.root_dir, opt.train_dir),
                os.path.join(opt.root_dir, opt.train_label),
                opt.train_less, opt.clean_data)
    elif opt.classes == 2:
        # 2 categories
        images, labels = utils.read_ice_snow_data(
                os.path.join(opt.root_dir, opt.train_dir),
                os.path.join(opt.root_dir, opt.train_label))

    # 7 categories
    # images, labels = utils.read_non_ice_snow_data(
    #         os.path.join(opt.root_dir, opt.train_dir),
    #         os.path.join(opt.root_dir, opt.train_label))

    ################ devide set #################
    if opt.fore:
        train_im, train_label = images[opt.num_val:], labels[opt.num_val:]
        val_im, val_label = images[:opt.num_val], labels[:opt.num_val]
    else:
        train_im, train_label = images[:-opt.num_val], labels[:-opt.num_val]
        val_im, val_label = images[-opt.num_val:], labels[-opt.num_val:]

    if opt.cu_mode:
        train_data_1 = train_im[:4439], train_label[:4439]
        train_data_2 = train_im[:5385], train_label[:5385]
        train_data_3 = train_im, train_label
        # train_datas = [train_data_1, train_data_2, train_data_3]
        train_datas = [train_data_2, train_data_3]
        opt.num_epochs //= len(train_datas)
    else:
        train_datas = [(train_im, train_label)]
    val_data = val_im, val_label
    #########################################

    if opt.retrain:
        state_dict = torch.load(
                opt.model_dir+'/'+opt.network+'-'+str(opt.layers)+'-'+str(opt.crop_size)+'_model.ckpt')
        model.load_state_dict(state_dict)

    ################ optimizer #################
    if opt.retrain and not opt.teacher_mode:
        if opt.network in ['effnet']:
            for param in model.module.parameters():
                param.requires_grad = False
            for param in model.module._fc.parameters():
                param.requires_grad = True
            # for param in model.module._swish.parameters():
            #     param.requires_grad = True
            for param in model.module.model._bn1.parameters():
                param.requires_grad = True

        elif opt.network in ['resnet', 'resnext', \
                             'resnext_wsl_32x8d', 'resnext_wsl_32x16d', 'resnext_wsl_32x32d', \
                             'resnext_swsl']:
            for param in model.parameters():
                param.requires_grad = False
            for param in model.module.fc.parameters():
                param.requires_grad = True
            for param in model.module.layer4[2].bn3.parameters():
            # for param in model.module.layer4[2].bn2.parameters():
                param.requires_grad = True

        elif opt.network in ['pnasnet_m', 'senet_m']:
            for param in model.module.parameters():
                param.requires_grad = False
            for param in model.module.classifier.parameters():
                param.requires_grad = True
            if opt.network == 'senet_m':
                for param in model.module.features.layer4.parameters():
                # for param in model.module.features.layer4[2].bn3.parameters():
                    param.requires_grad = True

        elif opt.network in ['inception_v3']:
            for param in model.parameters():
                param.requires_grad = False
            for param in model.fc.parameters():
                param.requires_grad = True
            for param in model.Mixed_7c.parameters():
                param.requires_grad = True
        else:
            for param in model.module.parameters():
                param.requires_grad = False
            for param in model.module.last_linear.parameters():
                param.requires_grad = True
            if opt.network in ['se_resnext50_32x4d', 'se_resnext101_32x4d']:
                for param in model.module.layer4[2].bn3.parameters():
                    param.requires_grad = True
            elif opt.network in ['senet154']:
                for param in model.module.layer4.parameters():
                    param.requires_grad = True
            elif opt.network in ['xception']:
                for param in model.module.bn4.parameters():
                    param.requires_grad = True
            elif opt.network in ['inceptionresnetv2']:
                for param in model.module.conv2d_7b.bn.parameters():
                    param.requires_grad = True
            elif opt.network in ['inceptionv4']:
                for param in model.module.features[-1].branch3.parameters():
                    param.requires_grad = True
            elif opt.network in ['fixpnas']:
                for param in model.module.cell_11.parameters():
                    param.requires_grad = True
                for param in model.module.cell_10.parameters():
                    param.requires_grad = True
                for param in model.module.cell_9.parameters():
                    param.requires_grad = True
        params = filter(lambda p: p.requires_grad, model.parameters())
    else:
        if opt.network in ['effnet'] and not opt.retrain:
            params = utils.add_weight_decay(model.module.model, 1e-4)
            params.append({'params': model.module._fc.parameters(), 'lr': opt.lr*10})
        else:
            params = utils.add_weight_decay(model, 1e-4)

    ################ optimizer #################
    optimizer = get_optimizer(opt, params, weight_decay=1e-4)
    if opt.scheduler in ['step', 'multistep', 'plateau', 'exponential']:
        scheduler = get_schedule(opt, optimizer)
    ############################################

    crop_size = opt.crop_size-128
    val_transforms = my_transform(False, crop_size)
    val_dataset = WeatherDataset(val_data[0], val_data[1], val_transforms)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=opt.batch_size,
                                             shuffle=False, num_workers=8, pin_memory=True)
    for train_data in train_datas:
        val_dis =np.bincount(val_label)+1e-20
        train_dis = np.bincount(train_data[1])
        print(val_dis, opt.num_val)
        print(train_dis, len(train_data[1]))

        train_transforms = my_transform(True, crop_size, opt.cutout, opt.n_holes, opt.length, opt.auto_aug, opt.rand_aug)
        train_dataset = WeatherDataset(train_data[0], train_data[1], train_transforms)

        train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=opt.batch_size,
                                               shuffle=True,
                                               num_workers=8,
                                               drop_last=True, pin_memory=True)

        loader = {'train':train_loader, 'val':val_loader}

        ################ scheduler #################
        if opt.scheduler in ['warmup', 'cycle', 'cos', 'cosw', 'sgdr']:
            scheduler = get_schedule(opt, optimizer, len(train_loader))
        ############################################

        model, acc = train_model(loader, model, criterion, optimizer, summary_writer,
                            scheduler=scheduler, scheduler_name=opt.scheduler, num_epochs=opt.num_epochs, device=device,
                            is_inception=opt.is_inception, mixup=opt.mixup, cutmix=opt.cutmix, alpha=opt.alpha,
                            val_dis=val_dis)
        utils.mkdir(opt.model_dir)
        torch.save(model.state_dict(),
            opt.model_dir+'/'+opt.network+'-'+str(opt.layers)+'-'+str(crop_size)+'_model.ckpt')

if __name__ == '__main__':
    opt = opts.parse_args()
    main(opt)











