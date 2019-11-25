import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    # Data input settings
    parser.add_argument('--root_dir', type=str, default='/home/lzw/datasets/air',
                    help='Path to root directory of datasets')
    parser.add_argument('--train_dir', type=str, default='train',
                    help='Path to training dataset')
    parser.add_argument('--test_dir', type=str, default='test',
                    help='Path to test dataset')
    parser.add_argument('--train_label', type=str, default='Train_label.csv',
                    help='Path to train labels')
    parser.add_argument('--val_label', type=str, default='val_label.csv',
                    help='Path to val labels')

    parser.add_argument('--model_dir', type=str, default='./model')
    parser.add_argument('--results_dir', type=str, default='./results')
    parser.add_argument('--log_dir', type=str, default='./log')
    parser.add_argument('--results_ts', type=str, default='./results_ts/')
    parser.add_argument('--res8', type=str, default='8-resnet-50-480_result.csv')

    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--seed', type=int, default=924)
    parser.add_argument('--crop_size', type=int, default=224, help='Training crop size')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--k_folds', type=int, default=6)
    parser.add_argument('--num_val', type=int, default=400)
    parser.add_argument('--num_epochs', type=int, default=120)
    parser.add_argument('--fore', type=int, default=1)
    parser.add_argument('--train_less', type=int, default=0)
    parser.add_argument('--clean_data', type=int, default=0)
    parser.add_argument('--weight', type=int, default=1)

    parser.add_argument('--optimizer', type=str, default='sgd',
                    help='sgd, adam, adamw, radam, novograd')
    parser.add_argument('--scheduler', type=str, default='multistep',
                    help='multistep, cycle, plateau, warmup')
    parser.add_argument('--lookahead', type=int, default=0)

    parser.add_argument('--cadene', type=int, default=0)
    parser.add_argument('--classes', type=int, default=9)
    parser.add_argument('--layers', type=int, default=101,
                    help='layer nums: 0-7, 18, 34, 50, 101, 152, 16, 19, 121, 161, 201, 48')
    parser.add_argument('--pretrained', type=int, default=1, help='pretrained 1=true, 0=false')
    parser.add_argument('--network', type=str, default='resnet',
        help='network: resnet, resnext, resnext_wsl(with battleneck_width arg), vgg, inception_v3')
    parser.add_argument('--battleneck_width', type=int, default=8, help='8, 16, 32, 48')
    parser.add_argument('--is_inception', type=int, default=0)
    parser.add_argument('--retrain', type=int, default=0)

    ########### mixup #################
    parser.add_argument('--mixup', type=int, default=0, help='use mixup could set alpha, cutmix')
    parser.add_argument('--alpha', type=float, default=1.0, help='for mixup alpha')
    parser.add_argument('--cutmix', type=int, default=0)

    ########## cutout #################
    parser.add_argument('--cutout', type=int, default=0, help='cutout need n_holes and length')
    parser.add_argument('--n_holes', type=int, default=1)
    parser.add_argument('--length', type=int, default=16)

    ########## auto_aug ###############
    parser.add_argument('--auto_aug', type=int, default=0)
    parser.add_argument('--rand_aug', type=int, default=0)

    ########## loss ##################
    parser.add_argument('--criterion', type=str, default='lsr', help='criterion: lsr(label smooth), focal, ce')
    parser.add_argument('--use_focal', type=int, default=0)
    ##### test #####
    parser.add_argument('--vote', type=int, default=0)
    parser.add_argument('--tta', type=int, default=1)

    ########## teacher #########
    parser.add_argument('--teacher_mode', type=int, default=0)
    parser.add_argument('--cu_mode', type=int, default=0)

    args = parser.parse_args()
    return args
