# train.py
#!/usr/bin/env	python3

""" train network using pytorch

author baiyu
"""

import os
import sys
import argparse
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter

from conf import settings
from utils import get_network, get_training_dataloader, get_test_dataloader, WarmUpLR, \
    most_recent_folder, most_recent_weights, last_epoch, best_acc_weights

# add horovod imports
import horovod.torch as hvd
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp

def train(epoch):

    start = time.time()
    net.train()
    for batch_index, (images, labels) in enumerate(cifar100_training_loader):

        if args.gpu:
            labels = labels.cuda()
            images = images.cuda()

        optimizer.zero_grad()
        outputs = net(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        n_iter = (epoch - 1) * len(cifar100_training_loader) + batch_index + 1

        last_layer = list(net.children())[-1]

        trained_samples = batch_index * args.b + len(images)
        # for name, para in last_layer.named_parameters():
        #     if 'weight' in name:
        #         writer.add_scalar('LastLayerGradients/grad_norm2_weights', para.grad.norm(), n_iter)
        #     if 'bias' in name:
        #         writer.add_scalar('LastLayerGradients/grad_norm2_bias', para.grad.norm(), n_iter)
        # if hvd.rank()==0:
        #     print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
        #         loss.item(),
        #         optimizer.param_groups[0]['lr'],
        #         epoch=epoch,
        #         trained_samples=batch_index * args.b + len(images),
        #         total_samples=len(cifar100_training_loader.dataset)
        #     ))

        #update training loss for each iteration
        # writer.add_scalar('Train/loss', loss.item(), n_iter)

        # if epoch <= args.warm:
        #     adjust_lr_warmup(epoch)
        #     # warmup_scheduler.step()

    # for name, param in net.named_parameters():
    #     layer, attr = os.path.splitext(name)
    #     attr = attr[1:]
    #     writer.add_histogram("{}/{}".format(layer, attr), param, epoch)

    finish = time.time()
    if hvd.rank()==0:
        print('Training Epoch: {epoch}\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
                loss.item(),
                optimizer.param_groups[0]['lr'],
                epoch=epoch
            ))  
        print('epoch {} training time consumed: {:.2f}s\tThroughput: {:.2f} imgs/sec/GPU'.format(epoch, finish - start, trained_samples / (finish - start)))
    return trained_samples / (finish - start)

@torch.no_grad()
def eval_training(epoch=0, tb=False):

    start = time.time()
    net.eval()

    test_loss = 0.0 # cost function error
    correct = 0.0

    for (images, labels) in cifar100_test_loader:

        if args.gpu:
            images = images.cuda()
            labels = labels.cuda()

        outputs = net(images)
        loss = loss_function(outputs, labels)

        test_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()

    finish = time.time()
    if args.gpu:
        if hvd.rank()==0:
            # print('GPU INFO.....')
            # print(torch.cuda.memory_summary(), end='')
            print('Evaluating Network.....')
            print('Test set: Epoch: {}, Average loss: {:.4f}, Accuracy: {:.4f}%, Time consumed:{:.2f}s'.format(
            epoch,
            test_loss / len(cifar100_test_loader.dataset),
            100.0 * (correct.float() / len(cifar100_test_loader.dataset)),
            finish - start
            ))
            print()

    #add informations to tensorboard
    if tb:
        writer.add_scalar('Test/Average loss', test_loss / len(cifar100_test_loader.dataset), epoch)
        writer.add_scalar('Test/Accuracy', correct.float() / len(cifar100_test_loader.dataset), epoch)

    return correct.float() / len(cifar100_test_loader.dataset)

def adjust_lr(epoch):
    if epoch < 60:
        lr_adj = 1.
    elif epoch < 120:
        lr_adj = 0.2
    elif epoch < 160:
        lr_adj = 0.2^2
    else:
        lr_adj *= 0.2^3
    for param_group in optimizer.param_groups:
        param_group['lr'] = args.lr * hvd.size() * lr_adj

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-gpu', action='store_true', default=False, help='use gpu or not')
    parser.add_argument('-b', type=int, default=128, help='batch size for dataloader')
    parser.add_argument('-warm', type=int, default=1, help='warm up training phase')
    parser.add_argument('-lr', type=float, default=0.1, help='initial learning rate')
    parser.add_argument('-resume', action='store_true', default=False, help='resume training')
    parser.add_argument('-epochs', type=int, default=200, help='epochs')
    args = parser.parse_args()

    # initialize horovod
    hvd.init()
    torch.manual_seed(42)

    if args.gpu:
        torch.cuda.set_device(hvd.local_rank())
        torch.cuda.manual_seed(42)

    cudnn.benchmark = True

    # Horovod: limit # of CPU threads to be used per worker.
    torch.set_num_threads(4)

    kwargs = {'num_workers': 4, 'pin_memory': True}
    # When supported, use 'forkserver' to spawn dataloader workers instead of 'fork' to prevent
    # issues with Infiniband implementations that are not fork-safe
    if (kwargs.get('num_workers', 0) > 0 and hasattr(mp, '_supports_context') and
            mp._supports_context and 'forkserver' in mp.get_all_start_methods()):
        kwargs['multiprocessing_context'] = 'forkserver'

    net = get_network(args)

    #data preprocessing:
    cifar100_training_loader = get_training_dataloader(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        batch_size=args.b,
        shuffle=False,
        **kwargs
    )

    cifar100_test_loader = get_test_dataloader(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        batch_size=args.b,
        shuffle=False,
        **kwargs
    )

    loss_function = nn.CrossEntropyLoss()

    if args.gpu:
        net = net.cuda()
        loss_function = loss_function.cuda()
    
    lr_scaler = hvd.size()
    
    optimizer = optim.SGD(net.parameters(), lr=(args.lr * lr_scaler), momentum=0.9, weight_decay=5e-4)
    # Horovod: wrap optimizer with DistributedOptimizer.
    optimizer = hvd.DistributedOptimizer(
        optimizer, named_parameters=net.named_parameters(),
        compression=hvd.Compression.none,
        op=hvd.Average
    )

    # train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=settings.MILESTONES, gamma=0.2) #learning rate decay
    iter_per_epoch = len(cifar100_training_loader)
    # warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)

    if args.resume:
        recent_folder = most_recent_folder(os.path.join(settings.CHECKPOINT_PATH, args.net), fmt=settings.DATE_FORMAT)
        if not recent_folder:
            raise Exception('no recent folder were found')

        checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder)

    else:
        checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, settings.TIME_NOW)

    #use tensorboard
    # if not os.path.exists(settings.LOG_DIR):
    #     os.mkdir(settings.LOG_DIR)

    #since tensorboard can't overwrite old values
    #so the only way is to create a new tensorboard log
    # writer = SummaryWriter(log_dir=os.path.join(
    #         settings.LOG_DIR, args.net, settings.TIME_NOW))
    # input_tensor = torch.Tensor(1, 3, 32, 32)
    # if args.gpu:
    #     input_tensor = input_tensor.cuda()
    # writer.add_graph(net, input_tensor)

    #create checkpoint folder to save model
    # if not os.path.exists(checkpoint_path):
    #     os.makedirs(checkpoint_path)
    # checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')

    best_acc = 0.0
    if args.resume:
        best_weights = best_acc_weights(os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder))
        if best_weights:
            weights_path = os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder, best_weights)
            print('found best acc weights file:{}'.format(weights_path))
            print('load best training file to test acc...')
            net.load_state_dict(torch.load(weights_path))
            best_acc = eval_training(tb=False)
            print('best acc is {:0.2f}'.format(best_acc))

        recent_weights_file = most_recent_weights(os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder))
        if not recent_weights_file:
            raise Exception('no recent weights file were found')
        weights_path = os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder, recent_weights_file)
        print('loading weights file {} to resume training.....'.format(weights_path))
        net.load_state_dict(torch.load(weights_path))

        resume_epoch = last_epoch(os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder))


    # Horovod: broadcast parameters & optimizer state.
    hvd.broadcast_parameters(net.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)

    img_sec_GPU = 0.0
    for epoch in range(1, args.epochs + 1):
        # if epoch > args.warm:
        #     adjust_lr(epoch)
        adjust_lr(epoch)

        if args.resume:
            if epoch <= resume_epoch:
                continue

        cur_tp = train(epoch)
        img_sec_GPU += cur_tp
        acc = eval_training(epoch)

        

        #start to save best performance model after learning rate decay to 0.01
        # if epoch > settings.MILESTONES[1] and best_acc < acc:
        #     weights_path = checkpoint_path.format(net=args.net, epoch=epoch, type='best')
        #     print('saving weights file to {}'.format(weights_path))
        #     torch.save(net.state_dict(), weights_path)
        #     best_acc = acc
        #     continue

        # if not epoch % settings.SAVE_EPOCH:
        #     weights_path = checkpoint_path.format(net=args.net, epoch=epoch, type='regular')
        #     print('saving weights file to {}'.format(weights_path))
        #     torch.save(net.state_dict(), weights_path)

    # writer.close()
    if hvd.rank() == 0:
        print("Average imgs per GPU: {:.2f}".format(img_sec_GPU/args.epochs))