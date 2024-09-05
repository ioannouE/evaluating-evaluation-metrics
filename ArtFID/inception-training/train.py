import argparse
import os
import torch
import torch.nn as nn
from torch.nn.functional import cross_entropy
import torchvision
from torchvision import transforms as T
import numpy as np
import os
import glob
import wandb
from tqdm import tqdm
from PIL import Image

import datasets
import networks
import inception
import lr_scheduler
import utils

import random


def main():
    # ----------
    # args
    # ----------
    parser = argparse.ArgumentParser()
    # Names and Paths
    parser.add_argument('--name', type=str, default='model')
    parser.add_argument('--group', type=str, default='art_classifier')
    parser.add_argument('--root_dir', type=str)
    parser.add_argument('--dataset_train_path', type=str)
    parser.add_argument('--dataset_val_path', type=str)
    parser.add_argument('--artist_class_weights_path', type=str)
    parser.add_argument('--style_class_weights_path', type=str)
    parser.add_argument('--class_weight_path', type=str)
    parser.add_argument('--pretrained_weights_path', type=str)
    parser.add_argument('--csv_file', type=str, default='output_dataset.csv')
    parser.add_argument('--dataset_dir', type=str, default='dataset')
    # Training
    parser.add_argument('--optim', type=str, default='adam', choices=['adam', 'sgd', 'rms'])
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--momentum', type=float, default=0.0) # SGD / RMSprop
    parser.add_argument('--dampening', type=float, default=0.0) # SGD
    parser.add_argument('--nesterov', action='store_true') # SGD
    parser.add_argument('--alpha', type=float, default=0.99) # RMSprop
    parser.add_argument('--centered', action='store_true') # RMSprop

    parser.add_argument('--pretrained', dest='pretrained', action='store_true')

    parser.add_argument('--scheduler', type=str, default='none', choices=['none', 'lambda_lr', 'cosine', 'exponential', 'cyclic', 'one_cycle'])
    parser.add_argument('--warmup_epochs', type=int, default=5)
    parser.add_argument('--scheduler_update_every', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.001)

    parser.add_argument('--aux_loss_weight', type=float, default=0.3)
    parser.add_argument('--label_smoothing', type=float, default=0.0)

    parser.add_argument('--num_steps', type=int, default=100000000)
    parser.add_argument('--batch_size', type=int, default=24)
    parser.add_argument('--gpus', nargs='*', type=int, default=None)
    parser.add_argument('--no_cuda', dest='cuda', action='store_false')
    parser.add_argument('--num_workers', type=int, default=8)
    # Logging
    parser.add_argument('--keep_ckpts', type=int, default=6)
    parser.add_argument('--log_freq', type=int, default=50)
    parser.add_argument('--save_freq', type=int, default=1000)
    parser.add_argument('--val_freq', type=int, default=1000)
    parser.add_argument('--no_wandb', dest='wandb', action='store_false')
    parser.set_defaults(wandb=False)
    parser.set_defaults(cuda=True)
    args = parser.parse_args()

    if args.gpus is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in args.gpus)

    # ----------
    # folders
    # ----------
    if not os.path.exists(os.path.join(args.root_dir, args.name)):
        os.makedirs(os.path.join(args.root_dir, args.name))

    args.checkpoint_dir = os.path.join(args.root_dir, args.name, 'checkpoints')
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)

    # ----------
    # logging
    # ----------
    if args.wandb:
        wandb.init(entity='your-username',
                   project='art-classifier-inception',
                   group=args.group,
                   config=args,
                   name=args.name,
                   dir=os.path.join(args.root_dir, args.name))

    train(args)


def train(args):
    # ----------
    # data
    # ----------e
    print('Create datasets...')

    train_transforms = T.Compose([
        T.Resize((299, 299)),
        # T.RandomCrop(size=(229, 229)),
        T.ToTensor(),
        T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
        T.RandomErasing(scale=(0.02, 0.33), p=0.5),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.RandomApply(torch.nn.ModuleList([T.GaussianBlur(kernel_size=(9, 9), sigma=(0.1, 2.0))]), p=0.2),
        #T.Normalize(mean=0.5, std=0.5),
    ])

    val_transforms = T.Compose([
        #T.CenterCrop(size=(256, 256)),
        T.ToTensor(),
        #T.Normalize(mean=0.5, std=0.5),
    ])

    import json
    # Load JSON file into a dictionary
    with open('labels_dicts/artists_3307_dict.json', 'r', encoding='utf8') as f:
        artist_label_dict = json.load(f)

    with open('labels_dicts/styles_96_dict.json', 'r', encoding='utf8') as f:
        style_label_dict = json.load(f)

    # artist_labels = collections.Counter(sample['artist_label'] for sample in train_dataset)
    num_artist_labels = 3307 # len(artist_labels)
    # unique_artist_labels = list(set(artist_labels))
    # num_artist_labels = len(unique_artist_labels)
    print(f"Number of unique artist labels: {num_artist_labels}")

    # style_labels = [sample['style_label'] for sample in train_dataset]
    # unique_style_labels = list(set(style_labels))
    num_style_labels = 96 # len(unique_style_labels)


    # Set a random seed for reproducibility
    torch.manual_seed(0)
    # Define the split ratio
    train_ratio = 0.8  # 80% for training, 20% for validation

   
    # train_dataset = datasets.H5Dataset(path_to_h5_file=args.dataset_train_path, transforms=train_transforms)
    dataset = datasets.CustomDataset(csv_file=args.csv_file, img_dir=args.dataset_dir, artist_label_dict=artist_label_dict, style_label_dict=style_label_dict, transforms=train_transforms)
    print('Dataset created')
    # Get the size of the dataset
    dataset_size = len(dataset)

    # Calculate the sizes of the splits
    train_size = int(train_ratio * dataset_size)
    val_size = dataset_size - train_size

    # Split the dataset
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

   
    print(f"Number of unique style labels: {num_style_labels}")
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    if args.artist_class_weights_path is not None:
        artist_class_weights = torch.from_numpy(np.load(args.artist_class_weights_path).astype(np.float32))
        style_class_weights = torch.from_numpy(np.load(args.artist_class_weights_path).astype(np.float32))
    else:
        artist_class_weights = torch.ones(num_artist_labels)
        artist_class_weights = artist_class_weights / artist_class_weights.sum() # normalize the weights so that they sum to 1
        style_class_weights = torch.ones(num_style_labels)
        style_class_weights = style_class_weights / style_class_weights.sum()

    # ----------
    # model
    # ----------
    print('Create model...')
    device = torch.device('cuda:0' if torch.cuda.is_available() and args.cuda else 'cpu')

    model = inception.Inception3(num_classes1=artist_class_weights.shape[0],
                                 num_classes2=style_class_weights.shape[0]).to(device)
    
    # Load pretrained imagenet weights as initialization
    if args.pretrained:
        state_dict = torch.load(args.pretrained_weights_path)
        model.load_state_dict(state_dict, strict=False)
    
    model.train()

    # ----------
    # optimizer
    # ----------
    if args.optim == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.optim == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, dampening=args.dampening, nesterov=args.nesterov)
    if args.optim == 'rms':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr, momentum=args.momentum, alpha=args.alpha, weight_decay=args.weight_decay, centered=args.centered)


    # ----------
    # schedule
    # ----------
    if args.scheduler == 'none':
        scheduler = None
    elif args.scheduler == 'cyclic':
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=args.lr, max_lr=0.001, step_size_up=2000, mode='triangular2')
    elif args.scheduler == 'one_cycle':
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, steps_per_epoch=9833, epochs=300)
    else:
        if args.scheduler == 'lambda_lr':
            scheduler2 = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 0.94 ** epoch)
        if args.scheduler == 'cosine':
            scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-10)
        if args.scheduler == 'exponential':
            scheduler2 = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.94)
        scheduler1 = lr_scheduler.LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=args.warmup_epochs)
        scheduler = lr_scheduler.SequentialLR(optimizer, schedulers=[scheduler1, scheduler2], milestones=[args.warmup_epochs])
    
    if args.scheduler in ['cyclic', 'one_cycle']:
        scheduler_update_every_step = True
    else:
        scheduler_update_every_step = False

    # ----------
    # training
    # ----------
    print('Training...')
    
    step = 0
    epoch = 0
    best_val_acc = 0.0
    last_artist_acc = 0.0
    last_style_acc = 0.0
    last_artist_aux_acc = 0.0
    last_style_aux_acc = 0.0

    # check if checkpoints are available
    if len(os.listdir(args.checkpoint_dir)) > 0:
        latest_ckpt = max(glob.glob(os.path.join(args.checkpoint_dir, '*.pth')), key=os.path.getctime)
        print(f'Loading latest checkpoint: {latest_ckpt}')
        try:
            checkpoint = torch.load(latest_ckpt, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if scheduler is not None:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            step = checkpoint['step']
            epoch = checkpoint['epoch']
            best_vall_acc = checkpoint['best_val_acc']
            print(f'Checkpoint loaded, starting at step {step}...')
        except:
            print('Could not load checkpoint, start training from stratch...')

    pbar = tqdm(total=args.num_steps)
    pbar.update(step)
    while step < args.num_steps:

        for batch_idx, batch in enumerate(train_loader):
            x = batch['image'].to(device)
            artist_label = batch['artist_label'].to(device)
            style_label = batch['style_label'].to(device)
            
            artist_pred, style_pred, artist_aux_pred, style_aux_pred = model(x)
        
            if torch.__version__.startswith('1.10'):
                artist_loss = cross_entropy(input=artist_pred, target=artist_label, weight=artist_class_weights.to(device), ignore_index=-1, reduction='sum', label_smoothing=args.label_smoothing)
                style_loss = cross_entropy(input=style_pred, target=style_label, weight=style_class_weights.to(device), ignore_index=-1, reduction='sum', label_smoothing=args.label_smoothing)

                artist_aux_loss = cross_entropy(input=artist_aux_pred, target=artist_label, weight=artist_class_weights.to(device), ignore_index=-1, reduction='sum', label_smoothing=args.label_smoothing)
                style_aux_loss = cross_entropy(input=style_aux_pred, target=style_label, weight=style_class_weights.to(device), ignore_index=-1, reduction='sum', label_smoothing=args.label_smoothing)
            else:
                artist_loss = cross_entropy(input=artist_pred, target=artist_label, weight=artist_class_weights.to(device), ignore_index=-1, reduction='sum')
                style_loss = cross_entropy(input=style_pred, target=style_label, weight=style_class_weights.to(device), ignore_index=-1, reduction='sum')

                artist_aux_loss = cross_entropy(input=artist_aux_pred, target=artist_label, weight=artist_class_weights.to(device), ignore_index=-1, reduction='sum')
                style_aux_loss = cross_entropy(input=style_aux_pred, target=style_label, weight=style_class_weights.to(device), ignore_index=-1, reduction='sum')
            
            loss = artist_loss + style_loss
            aux_loss = artist_aux_loss + style_aux_loss
            loss = loss + args.aux_loss_weight * aux_loss

            optimizer.zero_grad() 
            loss.backward()
            optimizer.step()

            # accuracy
            artist_acc = utils.compute_accuracy(artist_pred.detach(), artist_label)
            artist_acc = artist_acc if artist_acc != -1 else last_artist_acc
            artist_acc = float(artist_acc[0])
            artist_aux_acc = utils.compute_accuracy(artist_aux_pred.detach(), artist_label)
            artist_aux_acc = artist_aux_acc if artist_aux_acc != -1 else last_artist_aux_acc
            # artist_aux_acc = float(artist_aux_acc[0])

            style_acc = utils.compute_accuracy(style_pred.detach(), style_label)
            style_acc = style_acc if style_acc != -1 else last_style_acc
            style_acc = float(style_acc[0])
            style_aux_acc = utils.compute_accuracy(style_aux_pred.detach(), style_label)
            style_aux_acc = style_aux_acc if style_aux_acc != -1 else last_style_aux_acc
            
            if step % args.log_freq == 0:
                if args.wandb:
                    commit = not (step > 0 and step % args.val_freq == 0)
                    wandb.log({'loss/train/artist': artist_loss, 
                               'loss/train/style': style_loss,
                               'loss/train/style_aux': style_aux_loss,
                               'loss/train/artist_aux': artist_aux_loss,
                               'accuracy/train/artist_aux': artist_aux_acc,
                               'accuracy/train/style_aux': style_aux_acc,
                               'accuracy/train/artist': artist_acc,
                               'accuracy/train/style': style_acc,
                               'misc/optimizer/lr': args.lr if scheduler is None else scheduler.get_last_lr()[0]}, step=step, commit=commit)

            if step > 0 and step % args.val_freq == 0:
                artist_val_acc, style_val_acc = validate(model, device, args, val_loader)
                new_val_acc = (artist_val_acc + style_val_acc) / 2.0
                if args.wandb:
                    wandb.log({'accuracy/val/artist': artist_val_acc,
                               'accuracy/val/style': style_val_acc}, step=step)
                if new_val_acc > best_val_acc:
                    best_val_acc = new_val_acc
                    torch.save({'step': step,
                                'epoch': epoch,
                                'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
                                'best_val_acc': best_val_acc,
                                'num_classes_head1': artist_class_weights.shape[0],
                                'num_classes_head2': style_class_weights.shape[0]}, os.path.join(args.checkpoint_dir, f'checkpoint_{step}.pth'))

                    # Remove older checkpoints if number of checkpoints is larger than some specified number
                    if len(os.listdir(args.checkpoint_dir)) > args.keep_ckpts:
                        oldest_ckpt = min(glob.glob(os.path.join(args.checkpoint_dir, '*.pth')), key=os.path.getctime)
                        os.remove(oldest_ckpt)


            step += 1
            pbar.update(1)
            pbar.set_description(f'artist_acc: {artist_acc:.3f}, style_acc: {style_acc:.3f}')

            if scheduler is not None and scheduler_update_every_step: scheduler.step()
        
        if scheduler is not None and epoch % args.scheduler_update_every == 0 and not scheduler_update_every_step: scheduler.step()
        epoch += 1

    return best_val_acc


def validate(model, device, args, val_loader):
    val_transforms = T.Compose([
        #T.CenterCrop(size=(256, 256)),
        T.ToTensor(),
        #T.Normalize(mean=0.5, std=0.5),
    ])
    # val_dataset = datasets.H5Dataset(path_to_h5_file=args.dataset_val_path, transforms=val_transforms)
    # val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    model.eval()

    artist_val_acc = 0.0
    artist_val_count = 0
    style_val_acc = 0.0
    style_val_count = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            x = batch['image'].to(device)
            artist_label = batch['artist_label'].to(device)
            style_label = batch['style_label'].to(device)

            artist_pred, style_pred = model(x)
            
            artist_val_acc_ = utils.compute_accuracy(artist_pred, artist_label)
            if artist_val_acc_ != -1:
                artist_val_acc += artist_val_acc_[0]
                artist_val_count += 1 

            style_val_acc_ = utils.compute_accuracy(style_pred, style_label)
            if style_val_acc_ != -1:
                style_val_acc += style_val_acc_[0]
                style_val_count += 1 

    artist_val_acc /= artist_val_count
    style_val_acc /= style_val_count

    model.train()
    return artist_val_acc, style_val_acc


if __name__ == '__main__':
    main()
