import os
import random
from options import args_parser
from PIL import Image
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR#, StepLR
from torchvision.datasets import ImageFolder
import torchvision.models as models
import torchvision.transforms as transforms


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_datasets(args):
    train_dir = os.path.join(args.input_dir, args.train_dir)
    val_dir = os.path.join(args.input_dir, args.val_dir)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    if args.augment:
        train_aug = [
            transforms.Resize(256),
            transforms.TenCrop(args.img_size),  # TODO: play around with insert other augmentations here, like color jitter. and augmix (perhaps with imagenet policy)
            transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
            normalize,
        ]
        target_transform = [
            transforms.Lambda(lambda tgt: torch.full(size=(10, 1), fill_value=tgt))
        ]
    else:
        train_aug = [
            transforms.Resize(256),
            transforms.CenterCrop(args.img_size),
            transforms.ToTensor(),
            normalize,
        ]
        target_transform = []

    train_transforms = transforms.Compose(train_aug)
    target_transforms = transforms.Compose(target_transform)

    val_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(args.img_size),
        transforms.ToTensor(),
        normalize,
    ])

    # TODO: do a simple sanity check where we use cifar10 instead. if all is well we should get good convergence
    train_dataset = ImageFolder(train_dir, transform=train_transforms, target_transform=target_transforms)
    train_loader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    train_eval_dataset = ImageFolder(train_dir, transform=val_transforms)
    train_eval_loader = DataLoader(train_eval_dataset, batch_size=args.bs, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    val_dataset = ImageFolder(val_dir, transform=val_transforms)
    val_loader = DataLoader(val_dataset, batch_size=args.bs, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    assert len(train_dataset.classes) == len(val_dataset.classes), 'Number of classes in training set and validation set do not match!'
    return train_loader, train_eval_loader, val_loader, len(train_dataset.classes)


def get_model(arch, num_classes):
    if arch.lower() == 'resnet18':
        model = models.resnet18(num_classes=num_classes)
    else:
        raise ValueError('unkown architecture')
    return model


def eval_model(model, data_loader, args):
    model.eval()

    total = 0
    correct = 0
    with torch.no_grad():
        for (img, tgt) in data_loader:
            if args.augment:
                img = img.view(-1, 3, args.img_size, args.img_size)
                tgt = tgt.view(-1)

            img = img.to(args.device, non_blocking=True)
            tgt = tgt.to(args.device, non_blocking=True)
            outputs = model(img)
            _, pred = torch.max(outputs.data, dim=1)
            total += len(tgt)
            correct += (pred == tgt).sum().item()
    acc = (100 * correct / total)
    return acc


def main():
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    seed_everything(args.seed)

    save_dir = os.path.join(args.save_dir, args.train_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    else:
        assert args.overwrite, f'{save_dir} already exists but it not empty. If you want to overwrite the results, pass in the --overwrite argument'

    train_loader, train_eval_loader, val_loader, num_classes = get_datasets(args)
    model = get_model(args.arch, num_classes).to(args.device)
    criterion = nn.CrossEntropyLoss().to(args.device)

    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.wd)
    #scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    stats = {
        'epoch': [],
        'loss': [],
        'train_acc': [],
        'val_acc': [],
    }
    with tqdm(range(args.epochs), unit="Training Epoch") as tepoch:
        for epoch in tepoch:
            model.train()
            batch_loss = 0
            for i, (img, tgt) in enumerate(train_loader):
                optimizer.zero_grad()

                if args.augment:
                    img = img.view(-1, 3, args.img_size, args.img_size)
                    tgt = tgt.view(-1)

                img = img.to(args.device, non_blocking=True)
                tgt = tgt.to(args.device, non_blocking=True)
                output = model(img)
                loss = criterion(output, tgt)

                loss.backward()
                optimizer.step()
                batch_loss += loss.item()

            scheduler.step()

            train_acc = eval_model(model, train_eval_loader, args)
            val_acc = eval_model(model, val_loader, args)

            print(f'Epoch {epoch+1}: loss {batch_loss/(i+1):0.2f}, train acc {train_acc:0.2f}, val acc {val_acc:0.2f}')

            stats['epoch'].append(epoch + 1)
            stats['loss'].append(batch_loss / (i + 1))
            stats['train_acc'].append(train_acc)
            stats['val_acc'].append(val_acc)

    filename = os.path.join(save_dir, f'ckpt_ep={epoch+1:03d}_lr={args.lr}{"_augment" if args.augment else ""}.pt')
    torch.save({
            'stats': stats,
            'args': args,
            'model_state_dict': model.state_dict(),
            'opt_state_dict': optimizer.state_dict()
        },
        filename
    )
    print(f'Saving results in {filename}')


if __name__ == '__main__':
    main()













