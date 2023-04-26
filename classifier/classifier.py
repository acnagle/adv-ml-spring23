import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
from torch.optim.lr_scheduler import StepLR
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from options import args_parser
from tqdm import tqdm

def get_datasets(args):
    traindir = os.path.join(args.input_dir, 'train')
    valdir = os.path.join(args.input_dir, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    train_transforms = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])

    val_transforms = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])

    train_dataset = datasets.ImageFolder(traindir, transform = train_transforms)
    syn_val_dataset = datasets.ImageFolder(valdir, transform = val_transforms)
    cifar10_val_dataset = datasets.CIFAR10('data/cifar10', train=False, download=True, transform = val_transforms)

    return train_dataset, syn_val_dataset, cifar10_val_dataset

def get_model(model_type):
    model = torch.hub.load('pytorch/vision:v0.10.0', model_type, pretrained=True)
    return model

def test_model(model, args, dataset):

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.bs, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    total = 0
    correct = 0
    with torch.no_grad():
        for data in data_loader:
            images, labels = data
            outputs = model(images.to(args.device))
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.to(args.device)).sum().item()
    accuracy = (100 * correct / total)
    return accuracy



def main():
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    model = get_model(args.model)
    
    model.to(args.device)
    train_dataset, syn_val_dataset, cifar10_val_dataset = get_datasets(args)

    criterion = nn.CrossEntropyLoss().to(args.device)

    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.lr_decay)
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.bs, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    epoch_loss = []
    epoch_train_acc = []
    epoch_syn_val_acc = []
    epoch_cifar10_val_acc = []

    with tqdm(range(args.epochs), unit="Training Epoch") as tepoch:
        for epoch in tepoch:
            model.train()
            batch_loss = 0.0
            for i, (images, target) in enumerate(train_loader):
                images = images.to(args.device, non_blocking=True)
                target = target.to(args.device, non_blocking=True)
                output = model(images)
                loss = criterion(output, target)

                epoch_loss.append(loss.item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                batch_loss += loss.item()

            scheduler.step() 
            
            train_acc = test_model(model, args, train_dataset)
            syn_val_acc = test_model(model, args, syn_val_dataset)
            cifar10_val_acc = test_model(model, args, cifar10_val_dataset)
            
            print(batch_loss/(i+1), train_acc, syn_val_acc, cifar10_val_acc) 
            
            epoch_loss.append(batch_loss/(i+1))
            epoch_train_acc.append(train_acc)
            epoch_syn_val_acc.append(syn_val_acc)
            epoch_cifar10_val_acc.append(cifar10_val_acc)

    print('At the end of: {} epochs | Train accuracy of the model on the: {} images: {} %'.format(int(args.epochs) ,int(len(train_dataset.size)), float(epoch_train_acc[-1])))
    print('At the end of: {} epochs | Validation accuracy of the model on the: {} images: {} %'.format(int(args.epochs) ,int(len(syn_val_dataset.size)), float(epoch_syn_val_acc[-1])))
    print('At the end of: {} epochs | Validation accuracy of the model on the: {} images: {} %'.format(int(args.epochs) ,int(len(cifar10_val_dataset.size)), float(epoch_cifar10_val_acc[-1])))

if __name__ == '__main__':
    main()













