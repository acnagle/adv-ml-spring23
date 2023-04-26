import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='/home/an28622/data/bmw10_multi/', help="parent directory of training and validation images")
    parser.add_argument('--train_dir', type=str, default='train', help="name of the directory within args.input_dir containing the training data")
    parser.add_argument('--val_dir', type=str, default='val', help="name of the directory within args.input_dir containing the validation data")
    parser.add_argument('--save_dir', type=str, default='/home/an28622/adv-ml-spring23/results/', help="location for outputs" )
    parser.add_argument('--arch', type=str, default='resnet18', help="architecture of classifier" )
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID")
    parser.add_argument('--bs', type=int, default=128, help="batch size: B")
    parser.add_argument('--epochs', type=int, default=200, help="Total number of epochs")
    parser.add_argument('--lr', type=float, default=0.1, help="learning rate")
    parser.add_argument('--wd', type=float, default=1e-4, help="weight decay")
    parser.add_argument('--momentum', type=float, default=0.9, help="momentum")
    parser.add_argument('--num_workers', type=float, default=6, help="number of workers for data loader")
    parser.add_argument('--augment', action='store_true', help="use standard data augmentations")
    parser.add_argument('--img_size', type=int, default=224, help='size of input images passed into the model. any images larger than this size will be resized')
    parser.add_argument('--overwrite', action='store_true', help='overwrite any previous results')
    parser.add_argument('--seed', type=int, default=42, help='random seed')

    args = parser.parse_args()
    return args
