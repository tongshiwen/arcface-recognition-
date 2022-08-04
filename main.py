#coding:UTF-8
import os
import argparse
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from utils import progress_bar
from torch.autograd import Variable
from pd_load_imglist import ImageList
from model import FaceCNN
from backbone import Arcface
from torch.nn import DataParallel
from torch.optim.lr_scheduler import StepLR


parser = argparse.ArgumentParser(description='PyTorch FACE-RECOGNITION Training')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--start_epoch', default=0, type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--epochs', default=80, type=int,
                    help='number of total epochs to run')
parser.add_argument('--batch_size_temp', default=8, type=int, help='batchsizes for per epoch')
parser.add_argument('--root_path', default='', type=str, metavar='PATH',
                    help='path to root path of images (default: none)')
parser.add_argument('--num_classes', default=157995, type=int,
                    help='total of classes')
parser.add_argument('--train_list', default='', type=str, metavar='PATH',
                    help='path to training list (default: none)')
parser.add_argument('--val_list', default='', type=str, metavar='PATH',
                    help='path to validation list (default: none)')
args = parser.parse_args()

use_cuda = torch.cuda.is_available()
best_acc = 0  # best test accuracy

# Data
print('==> Preparing data..')
trainset=ImageList(root=args.root_path, fileList=args.train_list,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.428817), (0.259534)),
            ]))

trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size_temp,
                                          shuffle=True,
                                          num_workers=4, pin_memory=True)
def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight.data)
        torch.nn.init.constant(m.bias.data, 0.1)
        print('init!')
# Model
if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('')
    checkpoint = torch.load('')
    net = checkpoint['net']
    start_epoch = checkpoint['epoch']
else:
    print('==>Building model..')

    net = FaceCNN(num_classes = 10084)
    weights_init(net)

if use_cuda:
    net = torch.nn.DataParallel(net).cuda()
    cudnn.benchmark = True
print (net)
params = []
for name, value in net.named_parameters():
    if 'bias' in name:
        if 'fc1' in name or 'fc2' in name or 'fc3' in name:
            params += [{'params':value, 'lr': 2* args.lr, 'weight_decay': 0}]
        else:
            params += [{'params':value, 'lr': 2 * args.lr, 'weight_decay': 0}]
    else:
        if 'fc1' in name or 'fc2' in name or 'fc3' in name:
            params += [{'params':value, 'lr': 1 * args.lr}]
        else:
            params += [{'params':value, 'lr': 1 * args.lr}]
criterion = nn.CrossEntropyLoss()
def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if (epoch+1)%10 == 0:
        args.lr = args.lr *0.8
    if args.lr <= 0.00001:
        args.lr = 0.00001
    print ("---------------------->", args.lr )
    for param_group in optimizer.param_groups:
        param_group['lr'] = args.lr

metric_fc = Arcface(512, 10084)
metric_fc.cuda()
optimizer = torch.optim.SGD([{'params': net.parameters()}, {'params': metric_fc.parameters()}], lr=0.001, weight_decay=5e-4)
scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    train_loss = 0
    correct = 0
    total = 0
    index=0
    net.train()
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        targets = targets.long()
        inputs, targets = Variable(inputs), Variable(targets)
        targets = torch.squeeze(targets)
        outputs = net(inputs)
        outputs = metric_fc(outputs, targets)
        loss = criterion(outputs, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
        optimizer.step()
        train_loss += loss.data.item()
        if batch_idx %20==1:
            progress_bar(batch_idx, len(trainloader), 'Loss: %.3f, %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), loss.data.item(), 0., 0, total))
        # Save checkpoint.
        index+=1
        if index== int(len(trainloader)/2) or index== int(len(trainloader)):
            print('Saving..')
            state = {
                'net': net.module if use_cuda else net,
                'epoch': epoch,
            }
            if not os.path.isdir('./model'):
                os.mkdir('./model')
            if index== int(len(trainloader)/2):
                torch.save(state, './model/ckpt_1.t%d'%(epoch*2))
                torch.save(net.state_dict(), './model/resnetnew_1.pth%d'%(epoch*2))
            if index== int(len(trainloader)):
                torch.save(state, './model/ckpt_1.t%d'%(epoch*2+1))
                torch.save(net.state_dict(), './model/resnetnew_1.pth%d'%(epoch*2+1))
if __name__ == '__main__':
    for epoch in range(args.start_epoch, args.epochs):
        train(epoch)
        adjust_learning_rate(optimizer, epoch)