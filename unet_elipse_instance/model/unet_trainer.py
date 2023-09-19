import os
import time
from collections import OrderedDict
import torch
import torch.nn as nn
from model import UNet
import torch.optim as optim
import torch.nn.functional as F
from math import exp
import numpy as np
import cv2


# 计算一维的高斯分布向量
def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


# 4邻域的连通域和 8邻域的连通域
# [row, col]
NEIGHBOR_HOODS_4 = True
OFFSETS_4 = [[0, -1], [-1, 0], [0, 0], [1, 0], [0, 1]]

NEIGHBOR_HOODS_8 = False
OFFSETS_8 = [[-1, -1], [0, -1], [1, -1],
             [-1, 0], [0, 0], [1, 0],
             [-1, 1], [0, 1], [1, 1]]


def reorganize(binary_img: np.array):
    index_map = []
    points = []
    index = -1
    rows, cols = binary_img.shape
    for row in range(rows):
        for col in range(cols):
            var = binary_img[row][col]
            if var < 0.5:
                continue
            if var in index_map:
                index = index_map.index(var)
                num = index + 1
            else:
                index = len(index_map)
                num = index + 1
                index_map.append(var)
                points.append([])
            binary_img[row][col] = num
            points[index].append([row, col])
    return binary_img, points


def neighbor_value(binary_img: np.array, offsets, reverse=False):
    rows, cols = binary_img.shape
    label_idx = 0
    rows_ = [0, rows, 1] if reverse == False else [rows - 1, -1, -1]
    cols_ = [0, cols, 1] if reverse == False else [cols - 1, -1, -1]
    for row in range(rows_[0], rows_[1], rows_[2]):
        for col in range(cols_[0], cols_[1], cols_[2]):
            label = 256
            if binary_img[row][col] < 0.5:
                continue
            for offset in offsets:
                neighbor_row = min(max(0, row + offset[0]), rows - 1)
                neighbor_col = min(max(0, col + offset[1]), cols - 1)
                neighbor_val = binary_img[neighbor_row, neighbor_col]
                if neighbor_val < 0.5:
                    continue
                label = neighbor_val if neighbor_val < label else label
            if label == 255:
                label_idx += 1
                label = label_idx
            binary_img[row][col] = label
    return binary_img


# binary_img: bg-0, object-255; int
def Two_Pass(binary_img: np.array, neighbor_hoods):
    if neighbor_hoods == NEIGHBOR_HOODS_4:
        offsets = OFFSETS_4
    elif neighbor_hoods == NEIGHBOR_HOODS_8:
        offsets = OFFSETS_8
    else:
        raise ValueError

    binary_img = neighbor_value(binary_img, offsets, False)
    binary_img = neighbor_value(binary_img, offsets, True)

    return binary_img


def recursive_seed(binary_img: np.array, seed_row, seed_col, offsets, num, max_num=100):
    rows, cols = binary_img.shape
    binary_img[seed_row][seed_col] = num
    for offset in offsets:
        neighbor_row = min(max(0, seed_row + offset[0]), rows - 1)
        neighbor_col = min(max(0, seed_col + offset[1]), cols - 1)
        var = binary_img[neighbor_row][neighbor_col]
        if var < max_num:
            continue
        binary_img = recursive_seed(binary_img, neighbor_row, neighbor_col, offsets, num, max_num)
    return binary_img


# max_num 表示连通域最多存在的个数
def Seed_Filling(binary_img, neighbor_hoods, max_num=100):
    if neighbor_hoods == NEIGHBOR_HOODS_4:
        offsets = OFFSETS_4
    elif neighbor_hoods == NEIGHBOR_HOODS_8:
        offsets = OFFSETS_8
    else:
        raise ValueError

    num = 1
    rows, cols = binary_img.shape
    for row in range(rows):
        for col in range(cols):
            var = binary_img[row][col]
            if var <= max_num:
                continue
            binary_img = recursive_seed(binary_img, row, col, offsets, num, max_num=100)
            num += 1
    return binary_img


# 创建高斯核，通过两个一维高斯分布向量进行矩阵乘法得到
# 可以设定channel参数拓展为3通道
def create_window(window_size, channel=3):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def pca_loss(masks):
    num = 0
    mask_pred = masks.max(1)[1]

    # masks = softmax(masks)
    loss = 0
    mask_pred = mask_pred.detach().cpu().numpy()
    for mask in mask_pred:
        mask = (mask.astype(np.uint8)) * 255

        binary_img = Two_Pass(mask, NEIGHBOR_HOODS_8)
        binary_img, points = reorganize(mask)
        a = len(points)
        for i in range(a):
            num += 1
            poly0 = points[i]
            poly = np.array(poly0)
            poly1 = np.transpose(poly)
            mean = poly1.mean(axis=1)
            mean1 = mean.reshape((2, 1))
            x, y = poly1.mean(axis=1)
            poly2 = poly1 - mean1
            poly3 = np.transpose(poly2)
            m, n = poly.shape
            matrix = (1 / (m * n)) * (np.dot(poly2, poly3))
            # [s, v] = eig(Amatrix);
            s, v = np.linalg.eig(matrix)

            score = 0
            for j in range(m):
                dist = ((poly1[0][j] - x) * v[0][0] + (poly1[1][j] - y) * v[1][0]) ** 2 / (s[0] + 0.000001) + (
                        (poly1[0][j] - x) * v[1][0] + (poly1[1][j] - y) * v[1][1]) ** 2 / (s[1] + 0.000001)
                if dist < 1:
                    score += 1
            per = score / m
            loss += (1 - per)
    return loss / num

    # 计算SSIM
    # 直接使用SSIM的公式，但是在计算均值时，不是直接求像素平均值，而是采用归一化的高斯核卷积来代替。
    # 在计算方差和协方差时用到了公式Var(X)=E[X^2]-E[X]^2, cov(X,Y)=E[XY]-E[X]E[Y].
    # 正如前面提到的，上面求期望的操作采用高斯核卷积代替。


def ssim(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None):
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)

    if full:
        return ret, cs
    return ret


# Classes to re-use window
class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, val_range=None):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.val_range = val_range

        # Assume 1 channel for SSIM
        self.channel = 3
        self.window = create_window(window_size)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.dtype == img1.dtype:
            window = self.window
        else:
            window = create_window(self.window_size, channel).to(img1.device).type(img1.dtype)
            self.window = window
            self.channel = channel

        return ssim(img1, img2, window=window, window_size=self.window_size, size_average=self.size_average)


class FALoss(torch.nn.Module):
    def __init__(self, subscale=0.0625):
        super(FALoss, self).__init__()
        self.subscale = int(1 / subscale)

    def forward(self, feature1, feature2):
        feature1 = torch.nn.AvgPool2d(self.subscale)(feature1)
        feature2 = torch.nn.AvgPool2d(self.subscale)(feature2)

        m_batchsize, C, height, width = feature1.size()
        feature1 = feature1.view(m_batchsize, -1, width * height)  # [N,C,W*H]
        # L2norm=torch.norm(feature1,2,1,keepdim=True).repeat(1,C,1)   #[N,1,W*H]
        # # L2norm=torch.repeat_interleave(L2norm, repeats=C, dim=1)  #haven't implemented in torch 0.4.1, so i use repeat instead
        # feature1=torch.div(feature1,L2norm)
        mat1 = torch.bmm(feature1.permute(0, 2, 1), feature1)  # [N,W*H,W*H]

        m_batchsize, C, height, width = feature2.size()
        feature2 = feature2.view(m_batchsize, -1, width * height)  # [N,C,W*H]
        # L2norm=torch.norm(feature2,2,1,keepdim=True).repeat(1,C,1)
        # # L2norm=torch.repeat_interleave(L2norm, repeats=C, dim=1)
        # feature2=torch.div(feature2,L2norm)
        mat2 = torch.bmm(feature2.permute(0, 2, 1), feature2)  # [N,W*H,W*H]

        L1norm = torch.norm(mat2 - mat1, 1)

        return L1norm / ((height * width) ** 2)


class FocalLossSimple(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
        super(FocalLossSimple, self).__init__()
        self.cross_entropy = nn.CrossEntropyLoss(reduction='none')
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        bce_loss = self.cross_entropy(inputs, targets)
        pt = torch.exp(-bce_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss


class UNetTrainer(nn.Module):
    def __init__(self, opt, isTrain=True):
        super(UNetTrainer, self).__init__()
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.checkpoints_path = "checkpoints"
        self.isTrain = isTrain
        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names = ['seg']
        # specify the images you want to save/display. The program will call base_model.get_current_visual
        self.visual_names = ['img', 'gt_img', 'mask']
        # specify the models you want to save to the disk.
        # The program will call base_model.save_networks and base_model.load_networks
        self.model_names = ['UNet']
        self.netUNet = nn.DataParallel(UNet(opt.inchannel, opt.outchannel), opt.gpu_ids).cuda()
        self.softmax = nn.Softmax(dim=1)
        if self.isTrain:
            self.optimizer = optim.Adam(self.netUNet.parameters(),
                                        lr=opt.G_lr)
            self.optimizers = []
            self.optimizers.append(self.optimizer)
            self.schedulers = [optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.epochs) for optimizer in
                               self.optimizers]
            self.loss_function = nn.CrossEntropyLoss()
        self.print_networks(False)

    def forward(self, x):
        mask = self.netUNet(x)
        return self.softmax(mask)

    def set_input(self, img, gt_img):
        self.img = img.cuda()
        self.gt_img = gt_img.cuda()

    def optimize_parameters(self):
        mask = self.netUNet(self.img)
        gt_img = torch.squeeze(self.gt_img)
        # sr0 = torch.squeeze(sr)
        self.loss_seg = self.loss_function(mask, gt_img) + 0.1 * pca_loss(mask)
        self.mask = self.softmax(mask)
        self.optimizer.zero_grad()
        self.loss_seg.backward()
        self.optimizer.step()

    def isTrain(self, isTrain):
        self.isTrain = isTrain

    # update learning rate (called once every epoch)
    def update_learning_rate(self):
        for scheduler in self.schedulers:
            scheduler.step()
        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate = %.7f' % lr)

    # return visualization images. train.py will display these images, and save the images to a html
    def get_current_visuals(self):
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                if name == "gt_img":
                    visual_ret[name] = (getattr(self, name).detach()[0]).float() / 2.01
                elif name == 'mask':
                    visual_ret[name] = (getattr(self, name).detach()[0]).float()
                else:
                    visual_ret[name] = ((getattr(self, name).detach()[0] * 0.5 + 0.5).clamp(0, 1)).float()
        return visual_ret

    # return traning losses/errors. train.py will print out these errors as debugging information
    def get_current_losses(self):
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                # float(...) works for both scalar tensor and float number
                errors_ret['loss_' + name] = float(getattr(self, 'loss_' + name))
        return errors_ret

    # save models to the disk
    def save_networks(self, best_F):
        timestr = time.strftime('%m%d%H%M')
        save_dir = os.path.join(self.checkpoints_path, '%s_%.4f' % (timestr, best_F))
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)
        for name in self.model_names:
            if isinstance(name, str):
                save_filename = '%s_%.4f_net%s.pth' % (timestr, best_F, name)
                save_path = os.path.join(save_dir, save_filename)
                net = getattr(self, 'net' + name)

                if len(self.gpu_ids) > 1 and torch.cuda.is_available():
                    torch.save(net.module.state_dict(), save_path)
                else:
                    torch.save(net.state_dict(), save_path)
        return save_dir

    def load_networks(self, save_dir):
        which_model = os.path.split(save_dir)[1]
        for name in self.model_names:
            if isinstance(name, str):
                load_filename = '%s_net%s.pth' % (which_model, name)
                load_path = os.path.join(save_dir, load_filename)
                net = getattr(self, 'net' + name)
                if isinstance(net, torch.nn.DataParallel):
                    net = net.module
                if os.path.exists(load_path):
                    print('loading the model from %s' % load_path)
                    state_dict = torch.load(load_path)
                    net.load_state_dict(state_dict)
                else:
                    print('ignore the model %s' % 'net' + name)

    # print network information
    def print_networks(self, verbose):
        print('---------- Networks initialized -------------')
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
                if verbose:
                    print(net)
                print('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))
        print('-----------------------------------------------')
