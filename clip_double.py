import os
import clip
import torch
import argparse
import numpy as np
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader
from tqdm import tqdm
from data.open_set_datasets import get_class_splits, get_datasets
import utils
from utils import clip_cam, transformer_cam, compute_oscr
import copy
from sklearn.metrics import roc_auc_score
import pandas as pd
import coop
from coop import TransformerDecoderLayer 
from torch.cuda.amp import GradScaler, autocast
from torch.nn import functional as F
from dassl.metrics import compute_accuracy
import torch
import torch.nn as nn
from optim import build_optimizer, build_lr_scheduler
import train
from train import train_epoch, test, test_openset, test_double, test_openset_double, train_double
import random
from torch.optim.lr_scheduler import MultiStepLR
import torch.optim as optim
import cv2
from pytorch_grad_cam import GradCAM, \
    ScoreCAM, \
    GradCAMPlusPlus, \
    AblationCAM, \
    XGradCAM, \
    EigenCAM, \
    EigenGradCAM, \
    LayerCAM, \
    FullGrad
from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import show_cam_on_image, \
    deprocess_image, \
    preprocess_image
import pickle
#from data.imagenetv2 import ImageNetV2
#from data.data_manager import DatasetWrapper
from dassl.data import DatasetWrapper
from dassl.config import get_cfg_default
from data.augmentations import get_transform
from data.imagenet import get_image_net_datasets
from apex import amp

parser = argparse.ArgumentParser("Training")

# Dataset
parser.add_argument('--dataset', type=str, default='cub', help="")
parser.add_argument('--out-num', type=int, default=50, help='For cifar-10-100')
parser.add_argument('--image_size', type=int, default=224)

# optimization
parser.add_argument('--MAX_EPOCH', type=int, default=150)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--NAME', type=str, default='sgd')
parser.add_argument('--LR', type=float,  default=0.002)
parser.add_argument('--LR_SCHEDULER', type=str, default='cosine')
parser.add_argument('--WARMUP_EPOCH', type=int, default=1)
parser.add_argument('--WARMUP_TYPE', type=str, default='constant')
parser.add_argument('--WARMUP_CONS_LR', type=float,  default=1e-5)
parser.add_argument('--WEIGHT_DECAY', type=float,  default=5e-4)
parser.add_argument('--MOMENTUM', type=float,  default=0.9)
parser.add_argument('--SGD_DAMPNING', type=int, default=0)
parser.add_argument('--SGD_NESTEROV', type=utils.str2bool, default=False)
parser.add_argument('--RMSPROP_ALPHA', type=float,  default=0.99)
parser.add_argument('--ADAM_BETA1', type=float,  default=0.9)
parser.add_argument('--ADAM_BETA2', type=float,  default=0.999)
parser.add_argument('--STAGED_LR', type=utils.str2bool, default=False)
parser.add_argument('--NEW_LAYERS', type=tuple, default=())
parser.add_argument('--BASE_LR_MULT', type=float,  default=0.1)
parser.add_argument('--STEPSIZE', type=tuple,  default=(-1, ))
parser.add_argument('--GAMMA', type=float,  default=0.1)
parser.add_argument('--WARMUP_RECOUNT', type=utils.str2bool, default=True)
parser.add_argument('--WARMUP_MIN_LR', type=float,  default=1e-5)

# Eval
parser.add_argument('--eval', type=utils.str2bool, default=False)

# model
parser.add_argument('--beta', type=float, default=0.1, help="weight for entropy loss")
parser.add_argument('--model', type=str, default='classifier32')
parser.add_argument('--feat_dim', type=int, default=128, help="Feature vector dim, only for classifier32 at the moment")
# aug
parser.add_argument('--transform', type=str, default='rand-augment')
parser.add_argument('--rand_aug_m', type=int, default=None)
parser.add_argument('--rand_aug_n', type=int, default=None)
# misc
parser.add_argument('--num_workers', default=4, type=int)
parser.add_argument('--split_train_val', default=False, type=utils.str2bool,
                        help='Subsample training set to create validation set', metavar='BOOL')
parser.add_argument('--use_default_parameters', default=False, type=utils.str2bool,
                    help='Set to True to use optimized hyper-parameters from paper', metavar='BOOL')
parser.add_argument('--device', default='cuda:0', type=str, help='Which GPU to use')
parser.add_argument('--gpus', default=[0], type=int, nargs='+',
                        help='device ids assignment (e.g 0 1 2 3)')
parser.add_argument('--nz', type=int, default=100)
parser.add_argument('--ns', type=int, default=1)
parser.add_argument('--gpu', type=str, default='0')

parser.add_argument('--split_idx', default=0, type=int, help='0-4 OSR splits for each dataset')

#Prompt
parser.add_argument('--ctx_num', type=int, default=16)
parser.add_argument('--ctx_init', type=str, default='')
parser.add_argument('--csc', type=utils.str2bool, default=False)
parser.add_argument('--ctp', type=str, default='end')
parser.add_argument('--epoch', type=int, default=6)
parser.add_argument('--backbone', type=str, default='ViT-B/32')
parser.add_argument('--prec', type=str, default='fp16')  # fp16, fp32, amp

args = parser.parse_args()

#args.eval = True
device = "cuda" if torch.cuda.is_available() else "cpu"

#Save path

if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)

def reshape_transform(tensor, height=7, width=7):
    result = tensor[1:, :, :].reshape(height, width, tensor.size(1), tensor.size(2))
    result = result.permute(3,2,0,1)
    return result


#Prepare dataset
if args.dataset in ['svhn', 'cifar-10-10', 'cifar-10-100-10', 'cifar-10-100-50', 'tinyimagenet', 'imagenet_sketch', 'imagenet_v2', 'cifar-10-100']:

    args.train_classes, args.open_set_classes = get_class_splits(args.dataset, args.split_idx,
                                                                     cifar_plus_n=args.out_num)

    datasets = get_datasets(args.dataset, transform=args.transform, train_classes=args.train_classes,
                                open_set_classes=args.open_set_classes, balance_open_set_eval=True,
                                split_train_val=args.split_train_val, image_size=args.image_size, seed=args.seed,
                                args=args)

    if args.transform == 'rand-augment':
        if args.rand_aug_m is not None:
            if args.rand_aug_n is not None:
                datasets['train'].transform.transforms[0].m = args.rand_aug_m
                datasets['train'].transform.transforms[0].n = args.rand_aug_n

    target_classes = np.array(datasets['train'].classes)[args.train_classes].tolist()

elif args.dataset == 'tiny_imagenet_1k': 

    train_transform, test_transform = get_transform(transform_type=args.transform, image_size=args.image_size, args=args)

    datasets = get_image_net_datasets(train_transform=train_transform, test_transform=test_transform, train_classes=range(200),
                           open_set_classes=None, num_open_set_classes=1000, seed=0, osr_split='Easy')

    target_classes = np.array(datasets['train'].classes).tolist()

#for only svhn dataset:
if args.dataset == 'svhn':
    num_classes = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
    #num_classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    target_classes = np.array(num_classes)[args.train_classes].tolist()


#for only tinyimagenet dataset:
elif args.dataset in ['tinyimagenet' , 'tiny_imagenet_1k', 'imagenet_a', 'imagenet_sketch', 'imagenet_v2']:
    target_class_tmp=[]
    words_dict = {}
    words_name = pd.read_csv(os.path.join(args.save_path, word_txt), names=['indexs','name'], header=None, sep="\t")
    for i in words_name.index:
        words = words_name.loc[i].values
        words_dict[words[0]]=words[1]

    for i in target_classes:
        target_class_tmp.append(words_dict[i])
    target_classes = target_class_tmp
    
elif args.dataset == 'aircraft':
    for i in range(len(target_classes)):
        target_classes[i] = target_classes[i][:-1]

elif args.dataset == 'imagenet_1k':
    target_class_tmp=[]
    words_dict = {}
    with open(os.path.join(args.save_path, ILSVRC, classnames.txt)) as f:
        classnames = f.readlines()

    for i in classnames:
        words_dict[i[:9]]=i[10:-1]

    for i in target_classes:
        target_class_tmp.append(words_dict[i])
    target_classes = target_class_tmp
    args.train_classes = target_classes

args.target_classes = target_classes
print(target_classes)

#Prepare dataloader

dataloaders = {}
for k, v, in datasets.items():
    shuffle = True if k == 'train' else False
    dataloaders[k] = DataLoader(v, batch_size=args.batch_size,
                                shuffle=shuffle, sampler=None, num_workers=args.num_workers)


trainloader = dataloaders['train']
testloader = dataloaders['val']
outloader = dataloaders['test_unknown']

#Prepare text
text = torch.cat([clip.tokenize(f"a photo of a {c}") for c in target_classes]).to(device)

model_p, preprocess = clip.load(args.backbone, 'cpu')
model_p = coop.CustomCLIP(args, target_classes, model_p)
model_p = model_p.to(device)
for name, param in model_p.named_parameters():
    if "prompt_learner"  or 'multi_attention' in name:
        param.requires_grad_(True)
    else:
        param.requires_grad_(False)

#Optimize
optimizer = build_optimizer([{'params': model_p.prompt_learner.parameters()}, {'params': model_p.multi_attention.parameters()}], args)
#optimizer = build_optimizer(model_p, args)
scheduler = build_lr_scheduler(optimizer, args)

#model_p, optimizer = amp.initialize(model_p, optimizer, opt_level="O2")
#Dataparallel:
device_count = torch.cuda.device_count()
if device_count > 1:
    print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
    model_p = nn.DataParallel(model_p)
    #model_n = nn.DataParallel(model_n)


#Train:
#args.eval = True
if args.eval == False:
    for epoch in range(args.MAX_EPOCH):
        print('epoch:', epoch)
        train_double(model_p, optimizer, trainloader, epoch, args)
        scheduler.step()
        if (epoch+1) % 50 == 0:
            test(model_p, testloader, args)
            test_openset(model_p, testloader, outloader, args)
            print('Saving..')
            state = {
                'net_p': model_p.state_dict(),
                'epoch': epoch}
            torch.save(state, os.path.join(args.save_path, str(epoch) + 'split_'+ str(args.split_idx) + '.pth'))

else:
    checkpoint_p = torch.load(os.path.join(args.save_path, str(99)+ 'split_'+ str(args.split_idx) + '.pth'))
    model_p.load_state_dict(checkpoint_p['net_p'])
    test_openset(model_p, testloader, outloader, args)
    test(model_p, testloader, args)



