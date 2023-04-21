import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='/home/disha/Documents/lora/contents/cifar10/', help="location for input training images" )
    parser.add_argument('--model', type=str, default='resnet50', help="model type to use for classifier" )
    parser.add_argument('--output_dir', type=str, default='/home/disha/Documents/lora/contents/training_output/', help="location for outputs" )
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID")
    parser.add_argument('--bs', type=int, default=128, help="batch size: B")
    parser.add_argument('--epochs', type=int, default=200, help="Total number of epochs")
    parser.add_argument('--lr', type=float, default=0.1, help="learning rate")
    parser.add_argument('--lr_decay', type=float, default=1e-4, help="learning rate decay per round")
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--num_workers', default=4, type=float, help='number of workers for data loader')

    args = parser.parse_args()
    return args
