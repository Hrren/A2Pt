from torch.cuda.amp import GradScaler, autocast
from dassl.metrics import compute_accuracy
import torch
import torch.nn as nn
from optim import build_optimizer, build_lr_scheduler
from torch.nn import functional as F
import copy
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import os
import clip
import torch
import argparse
import numpy as np
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from coop import LayerNorm
from tensorboardX import SummaryWriter
from utils import compute_oscr
from apex import amp

writer = SummaryWriter('tens/')


device = "cuda" if torch.cuda.is_available() else "cpu"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

#model_tmp, preprocess = clip.load('ViT-B/32', 'cuda:0')

def compute_logits(image_features, text_features):
    ln_post = LayerNorm(512)
    image_features = ln_post(image_features[:, 0, :])

    b, _ = image_features.size()
    text_features = text_features[:20, :]
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    logits = image_features @ text_features.t()

    return logits

class FocalLoss(nn.Module):
    r"""
        This criterion is a implemenation of Focal Loss, which is proposed in 
        Focal Loss for Dense Object Detection.

            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])

        The losses are averaged across observations for each minibatch.

        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5), 
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.


    """
    def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs,dim=1)

        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)



        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]

        probs = (P*class_mask).sum(1).view(-1,1)

        log_p = probs.log()


        batch_loss = -alpha*(torch.pow((1-probs), self.gamma))*log_p 


        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss

focal_loss = FocalLoss(20)

def draw_picture(act_close, act_open):
    plt.style.use('classic')
    n, bins, patches = plt.hist(act_close, 50, facecolor='green', alpha=0.5, histtype='bar', label='unknown', edgecolor='white')
    n, bins, patches = plt.hist(act_open, 50, facecolor='blue', alpha=0.5, histtype='bar', label='unknown', edgecolor='white')
    plt.grid(False)
    plt.close()

def class_dis(target_close, pred_close, pred_open, act_close, act_open, class_want):
    pred_close = torch.tensor([item.cpu().detach().numpy() for item in pred_close])
    pred_open = torch.tensor([item.cpu().detach().numpy() for item in pred_open])

    act_close_pred_tmp = np.reshape(np.array(pred_close), (-1))
    act_open_pred = np.reshape(np.array(pred_open), (-1))


    label_want_idx_close = []
    label_want_idx_open = []


    for i, j in enumerate(act_open_pred):
        if j == class_want:
            label_want_idx_open.append(i)


    act_open = act_open[label_want_idx_open]

    print(act_open.shape)


    idx_open = []
    for i, j in enumerate(act_open):
        if j >= 15:
            idx_open.append(i)
    for i in idx_open:
        print(label_want_idx_open[i], act_open_pred[label_want_idx_open[i]])

def test(model, testloader, args):
    model.eval()
    output_list = []
    out_label = []
    loss_list = []
    t = 0
    
    with torch.no_grad():
        for batch_idx, (image, label, idx) in enumerate(tqdm(testloader)):
            t = batch_idx
            image, label = image.to(device), label.to(device)
            out_label.append(label)
            output, _, _, _ = model(image)
            
            output_list.append(output)
            out = torch.cat(output_list)

    labels = torch.cat(out_label)
    acc = compute_accuracy(out, labels)[0].item()
    print('acc:', acc)
    

def test_openset(model, testloader, outloader, args):
    model.eval()
    act_close = []
    target_close = []
    pred_close = []
    with torch.no_grad():
        for batch_idx, (images, labels, idx) in enumerate(tqdm(testloader)):
            images, labels = images.to(device), labels.to(device)
            target_close.append(labels)
            logits_per_image, _, _, _  = model(images)
            
            values, indices = torch.max(logits_per_image, dim=1)
            pred_close.append(indices)
            prob = F.softmax(logits_per_image,dim=1)
            for i in range(prob.size(0)):
                values, indices_pre = torch.topk(logits_per_image[i], 1, dim=0, largest=True, sorted=True, out=None)
                act = values[0].unsqueeze(0)
                act_close.append(act)
        act_close = np.reshape(np.array(torch.cat(act_close).cpu()), (-1))    
        target_close = np.reshape(np.array(torch.cat(target_close).cpu()), (-1))
        pred_close = np.reshape(np.array(torch.cat(pred_close).cpu()), (-1))

    act_open = []
    target_open = []
    pred_open = []
    with torch.no_grad():
        for batch_idx, (images, labels, idx) in enumerate(tqdm(outloader)):
            images, labels = images.to(device), labels.to(device)
            target_open.append(labels)
            logits_per_image, _, _, _ = model(images)
            values, indices = torch.max(logits_per_image, dim=1)
            pred_open.append(indices)
            prob = F.softmax(logits_per_image,dim=1)
            for i in range(logits_per_image.size(0)):
                values, indices_pre = torch.topk(logits_per_image[i], 1, dim=0, largest=True, sorted=True, out=None)
                act = values[0].unsqueeze(0)
                act_open.append(act)
        act_open = np.reshape(np.array(torch.cat(act_open).cpu()), (-1))
        target_open = np.reshape(np.array(torch.cat(target_open).cpu()), (-1))
        target_open[:] = len(args.target_classes)

    pred_open = pred_open[:-1]

    act_close_solo = copy.deepcopy(act_close)
    act_open_solo = copy.deepcopy(act_open)
    auc_pred_solo = np.hstack([act_close_solo, act_open_solo])

    auc_known_labels = copy.deepcopy(target_close)
    auc_unknown_labels = copy.deepcopy(target_open)
    auc_known_labels[:] = 1
    auc_unknown_labels[:] = 0
    auc_labels = np.hstack([auc_known_labels, auc_unknown_labels])
    auc = roc_auc_score(auc_labels, auc_pred_solo)
    print('auc', auc)
    oscr = compute_oscr(act_close_solo, act_open_solo, pred_close, target_close)
    print('oscr', oscr)

def train_double(model_p, optimizer, dataloader, epoch, args):

    output_list = []
    out_label = []
    loss_list = []
    loss_sim = []
    loss_pos_list = []
    loss_neg_list = []
    t = 0
    for batch_idx, (image, label, idx) in enumerate(tqdm(dataloader)):
        t = batch_idx
        image, label = image.to(device), label.to(device)
        out_label.append(label)
        if args.prec == "amp":
            scaler = GradScaler()
            with autocast():
                output = model(image)
                loss = F.cross_entropy(output, label)
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            output_for, output_back, output_com, sim_fb = model_p(image)
            output_list.append(output_for)

            #for not prompt loss
                       
            target_en = torch.Tensor(label.shape[0], len(args.target_classes)).to(device)
            target_en.zero_()
            target_en.scatter_(1, label.view(-1, 1), 1)
            soft_out = F.softmax(output_back,dim=1)
            exp_soft_out = torch.exp(soft_out)
            log_soft_out = torch.log(soft_out)
            
            loss_neg = 0.01 * (-1 * log_soft_out).sum() /  output_back.size(0) #- F.nll_loss(exp_soft_out, label)
            loss_pos = F.cross_entropy(output_for, label)
            sim_fb_loss = sim_fb.sum() / sim_fb.size(0)
            sim_fb_loss = 1 / (-1 * sim_fb_loss)
            # for normal loss
            loss = loss_pos + sim_fb_loss + 21 * loss_neg
            loss_list.append(loss.item())
            loss_pos_list.append(loss_pos.item())
            loss_neg_list.append(loss_neg.item())
            loss_sim.append(sim_fb_loss.item())
            
            loss_list.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    out = torch.cat(output_list)
    labels = torch.cat(out_label)
    loss = sum(loss_list) / (t + 1)
    loss_pos = sum(loss_pos_list) / (t + 1)
    loss_neg = sum(loss_neg_list) / (t + 1)
    loss_sim = sum(loss_sim) / (t + 1)
    writer.add_scalar('loss_pos', loss_pos, epoch)
    writer.add_scalar('loss_neg', loss_neg, epoch)
    writer.add_scalar('loss_sim', loss_sim, epoch)
    writer.add_scalar('loss_total', loss, epoch)
    acc = compute_accuracy(out, labels)[0].item()
    print('acc:', acc)
    print('loss', loss)


