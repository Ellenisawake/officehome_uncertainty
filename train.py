import os
import numpy as np
import cv2
from datetime import datetime
import time
import math

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from torch.utils.data.sampler import SequentialSampler, RandomSampler
from torch.distributions.normal import Normal
import torch.optim as optim

from resnet_aap import resnet50backbone


class OfficeHomeReducedDataset(Dataset):
    num_class = 65

    def __init__(self, dataset_root, domain, mode='train', transforms=None):
        super().__init__()
        self.transforms = transforms
        self.origin_dir = dataset_root + '/OfficeHomeDataset_10072016'
        self.domain = domain
        self.mode = mode
        self.names = np.loadtxt(self.origin_dir + '/categories.txt', dtype=str)
        images_list, labels = [], []
        images_list.extend(np.loadtxt(self.origin_dir + '/%s_images.txt' % domain, dtype=str))
        labels.extend(np.loadtxt(self.origin_dir + '/%s_labels.txt' % domain, dtype='i8'))
        reduced_ids = np.loadtxt(self.origin_dir + '/reduced1024_%s_ids.txt' % domain, dtype='i8')
        self.labels = np.asarray(labels)[reduced_ids]
        self.img_list = np.asarray(images_list)[reduced_ids]

    def get_img(self, index):
        img_file = self.origin_dir + '/' + self.img_list[index]
        img = cv2.imread(img_file)
        return img

    def get_label(self, index):
        label = self.labels[index]
        return label

    def __getitem__(self, index):
        img_origin = self.get_img(index)
        label = self.get_label(index)
        if self.transforms is not None:
            img_origin = self.transforms(img_origin)
        return img_origin, label, index

    def __len__(self):
        return len(self.img_list)


# multiplied sigmoid value to be be in range(0, 2). centered around 1.0 when sigma=0.0
class ResAFNAleatoricClassifierV2(nn.Module):
    def __init__(self, channels=(2048, 256, 12), nu_sigmoid=2.0, extract=True, dropout_p=0.5):
        super().__init__()
        self.fc1_1 = nn.Sequential(
            nn.Linear(channels[0], channels[1]),  # bias=True???
            nn.BatchNorm1d(channels[1], affine=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p))
        self.fc1_2 = nn.Sequential(
            nn.Linear(channels[0], channels[1]),  # bias=True???
            nn.BatchNorm1d(channels[1], affine=True))
        self.fc2 = nn.Linear(channels[1], channels[-1])
        self.extract = extract
        self.dropout_p = dropout_p
        self.dropout = nn.Dropout(p=dropout_p)
        self.nu_sigmoid = nu_sigmoid

    def forward_inference(self, x):
        fc1_emb = self.fc1_1(x)
        logit = self.fc2(fc1_emb)
        return logit

    def forward_with_sigma(self, x):
        mu = self.fc1_1(x)
        sigma = self.fc1_2(x)
        logit = self.fc2(mu)
        return logit, sigma

    def forward_train_sigmoid(self, x, epsilon):
        mu = self.fc1_1(x)
        sigma = self.nu_sigmoid*torch.sigmoid(self.fc1_2(x))
        mu.mul_(math.sqrt(1 - self.dropout_p))
        logit = self.fc2(mu + epsilon * sigma)
        return logit, mu, sigma


def log_with_print(writer, text):
    print(text)
    writer.write('%s\n' % text)


# adapted from SAFN code ---------------------------------------------------------
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.01)
        m.bias.data.normal_(0.0, 0.01)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.01)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.01)
        m.bias.data.normal_(0.0, 0.01)


def get_L2norm_loss_self_driven(x):
    radius = x.norm(p=2, dim=1).detach()
    assert radius.requires_grad == False
    radius = radius + 1.0
    l = ((x.norm(p=2, dim=1) - radius) ** 2).mean()
    return l
# adapted from SAFN code ---------------------------------------------------------


def train(args):
    # set cuda device ----------------------
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device  # str

    today = str(datetime.date(datetime.now()))
    year_month_date = today.split('-')
    date_to_save = year_month_date[0][2:] + year_month_date[1] + year_month_date[2]
    # ------------------------------------------------------------------------------
    save_dir = args.save_dir + '/%s_results_reducedofficehome' % date_to_save
    # root directory in which OfficeHomeDataset_10072016 is placed in
    dataset_root = args.data_dir
    # use pytorch official ImageNet pre-trained model
    resnet_model_dir = args.resnet_model_dir+ '/resnet50-19c8e357.pth'
    # ------------------------------------------------------------------------------

    init_rate = args.lr  # 0.001
    num_epochs = args.num_epoch
    batch_size = args.batch_size
    cls_lr = args.cls_lr
    fc_dim = args.fc_dim
    source = args.source
    target = args.target
    w_l2 = args.weight_L2norm
    w_kl = args.weight_kl
    nu_sigmoid = args.nu_sigmoid
    tgt_keep = args.tgt_keep

    exp_name = 'norm_alea_tgtkeep%.1f_' % (tgt_keep) + source[0] + '2' + target[0]
    save_log_file = save_dir + '/train_%s.txt' % (exp_name)
    os.makedirs(save_dir) if not os.path.exists(save_dir) else None
    img_resize, img_size = 256, 224
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    train_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(img_resize),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(img_size),
        # augmentation on the source samples
        # transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=None, shear=10),
        # transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)])
    test_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(img_resize),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)])
    train_src_set = OfficeHomeReducedDataset(dataset_root, source, transforms=train_transforms)
    train_tgt_set = OfficeHomeReducedDataset(dataset_root, target, transforms=train_transforms)
    test_set = OfficeHomeReducedDataset(dataset_root, target, transforms=test_transforms)
    train_size, test_size = len(train_src_set), len(test_set)
    num_class = test_set.num_class
    num_batches = min(train_size // batch_size, test_size // batch_size)
    train_src_loader = DataLoader(train_src_set,
                               sampler=RandomSampler(train_src_set),
                               batch_size=batch_size,
                               drop_last=True,
                               num_workers=2,
                               pin_memory=True)
    train_tgt_loader = DataLoader(train_tgt_set,
                               sampler=RandomSampler(train_tgt_set),
                               batch_size=batch_size,
                               drop_last=True,
                               num_workers=2,
                               pin_memory=True)
    test_loader = DataLoader(test_set,
                               sampler=SequentialSampler(test_set),
                               batch_size=batch_size,
                               drop_last=False,
                               num_workers=2,
                               pin_memory=True)

    # net -----------------------------------------------------------------------------
    netG = resnet50backbone(pre_trained=resnet_model_dir)
    netF = ResAFNAleatoricClassifierV2(channels=(2048, fc_dim, num_class), nu_sigmoid=nu_sigmoid)
    best_state_dict_G = netG.state_dict()
    best_state_dict_F = netF.state_dict()
    netG.cuda().train()
    netF.cuda().train()
    netF.apply(weights_init)  # classifier initialization as in original AFN code
    opt_g = optim.SGD(netG.parameters(), lr=init_rate, weight_decay=0.0005)
    opt_f = optim.SGD(netF.parameters(), lr=init_rate*cls_lr, momentum=0.9, weight_decay=0.0005)
    criterion_cls = nn.CrossEntropyLoss().cuda()

    # --------------------------------------------------------------------
    l_ce_val, l_norm_val, l_s_kl_val, l_t_kl_val, best_acc = 0.0, 0.0, 0.0, 0.0, 0.0
    bs_keep = int(batch_size*tgt_keep)
    log_writer = open(save_log_file, 'w')
    time_now = str(datetime.time(datetime.now()))[:8]
    log_with_print(log_writer, 'Start time: %s (Date: %s)' % (time_now, date_to_save))
    log_with_print(log_writer, '***************************')
    log_with_print(log_writer, 'Exp: %s (on gpu%s)' % (exp_name, args.device))
    log_with_print(log_writer, '***************************')
    log_with_print(log_writer, 'Data directory: %s' % dataset_root)
    log_with_print(log_writer, 'Save directory: %s' % save_dir)
    log_with_print(log_writer, 'learning rate G: %f' % init_rate)
    log_with_print(log_writer, 'learning rate F: %f' % (init_rate*cls_lr))
    log_with_print(log_writer, 'Weight of L2Norm: %f' % w_l2)
    log_with_print(log_writer, 'Weight of KLD: %f' % w_kl)
    log_with_print(log_writer, 'nu_sigmoid: %f' % nu_sigmoid)
    log_with_print(log_writer, 'fc_dim: %d' % fc_dim)
    log_with_print(log_writer, 'tgt_keep: %d(of%d)' % (bs_keep, batch_size))
    log_with_print(log_writer, 'Dataset size: %d %d\tBatch size: %d' % (train_size, test_size, batch_size))
    log_with_print(log_writer, 'Start training on %d batches...' % (num_batches))
    log_writer.flush()
    s_epsilons = Normal(torch.tensor([0.0]), torch.tensor([1.0]))
    src_sigma, tgt_sigma = 0.0, 0.0
    start = time.time()
    for epoch in range(1, num_epochs+1):
        it = 1
        netG.train()
        netF.train()
        for (img_src, label_s, _), (img_tgt, _, _) in zip(train_src_loader, train_tgt_loader):
            img_src, label_s = img_src.cuda(), label_s.cuda()
            img_tgt = img_tgt.cuda()
            opt_g.zero_grad()
            opt_f.zero_grad()
            epsilons = s_epsilons.sample(sample_shape=torch.Size([batch_size*2])).cuda()
            s_logit, s_mus, s_sigmas = netF.forward_train_sigmoid(netG(img_src), epsilons[:batch_size])
            t_logit, t_mus, t_sigmas = netF.forward_train_sigmoid(netG(img_tgt), epsilons[batch_size:])
            l_ce = criterion_cls(s_logit, label_s)
            l_norm = w_l2 * (get_L2norm_loss_self_driven(s_mus) + get_L2norm_loss_self_driven(t_mus))
            l_s_kl = 0.5 * (s_mus ** 2 + s_sigmas ** 2 - 2.0 * torch.log(s_sigmas) - 1.0).mean()
            if tgt_keep > 0.0:
                t_stds = t_sigmas.mean(dim=1).detach()
                ids = t_stds.sort(descending=False)
                t_mus = t_mus[ids[1][:bs_keep], :]
                t_sigmas = t_sigmas[ids[1][:bs_keep], :]
                l_t_kl = 0.5 * (t_mus ** 2 + t_sigmas ** 2 - 2.0 * torch.log(t_sigmas) - 1.0).mean()
                l_total = l_ce + l_norm + w_kl * (l_s_kl+l_t_kl)
                l_t_kl_val = float(l_t_kl.item())
            else:
                l_total = l_ce + l_norm + w_kl * l_s_kl
            l_total.backward()
            opt_g.step()
            opt_f.step()
            if it == num_batches:
                l_ce_val = float(l_ce.item())
                l_norm_val = float(l_norm.item())
                l_s_kl_val = float(l_s_kl.item())
                # l_t_kl_val = float(l_t_kl.item())
                src_sigma = float(s_sigmas.mean().item())
                tgt_sigma = float(t_sigmas.mean().item())
                break
            it += 1
        netF.eval()
        netG.eval()
        correct = 0.0
        with torch.no_grad():
            for image, label, _ in test_loader:
                image, label = image.cuda(), label.cuda()
                output = netF.forward_inference(netG(image))
                pred = output.data.max(1)[1]
                correct += float(pred.eq(label.data).cpu().sum())
        acc_ep = float(correct) / float(test_size) * 100.0
        if acc_ep > best_acc:
            best_acc = acc_ep
            best_state_dict_G = netG.cpu().state_dict()
            best_state_dict_F = netF.cpu().state_dict()
            netG.cuda()
            netF.cuda()
        time_taken = (time.time() - start) / 60.0
        log_with_print(log_writer, 'epoch%02d: l_ce:%f l_norm:%.2f l_s_kl:%f l_t_kl:%f s_sigma:%.4f t_sigma:%.4f test_acc:%.2f/%.2f in %.1fmin' % (
            epoch, l_ce_val, l_norm_val, l_s_kl_val, l_t_kl_val, src_sigma, tgt_sigma, acc_ep, best_acc, time_taken))
        log_writer.flush()
    time_taken = (time.time() - start) / 3600.0
    time_now = str(datetime.time(datetime.now()))[:8]
    log_with_print(log_writer, 'Finish time: %s (Date: %s)' % (time_now, date_to_save))
    torch.save(best_state_dict_G, save_dir + '/%s_best_%.2f_netG.pth' % (exp_name, best_acc))
    torch.save(best_state_dict_F, save_dir + '/%s_best_%.2f_netF.pth' % (exp_name, best_acc))
    log_with_print(log_writer, '\nBest accuracy: %.2f' % best_acc)
    log_with_print(log_writer, 'Total time taken: %.1f hrs' % time_taken)
    log_writer.close()
    print('Finish training')


if __name__ == '__main__':
    RUN_FILE = os.path.basename(__file__)
    print('running %s...' % RUN_FILE)
    # check_dataset()
    import argparse
    parser = argparse.ArgumentParser(description='OfficeHome DA')
    parser.add_argument('--device', type=str, default='0', metavar='TS', help='Domain ID')
    parser.add_argument('--data_dir', type=str, default='/home/jian/datasets', metavar='TS', help='Domain ID')
    parser.add_argument('--save_dir', type=str, default='/home/jian/results', metavar='TS', help='Domain ID')
    parser.add_argument('--resnet_model_dir', type=str, default='/home/jian/models', metavar='TS', help='Domain ID')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='Domain ID')
    parser.add_argument('--num_epoch', type=int, default=60, metavar='NE', help='Domain ID')
    parser.add_argument('--batch_size', type=int, default=32, metavar='BS', help='Domain ID')
    parser.add_argument('--source', type=str, default='Product', metavar='SR', help='Domain ID')
    parser.add_argument('--target', type=str, default='RealWorld', metavar='TS', help='Domain ID')
    parser.add_argument("--fc_dim", type=int, default=512)
    parser.add_argument("--weight_L2norm", type=float, default=0.05)
    parser.add_argument("--weight_kl", type=float, default=0.01)
    parser.add_argument("--cls_lr", type=float, default=10.0)
    parser.add_argument("--nu_sigmoid", type=float, default=2.0)
    parser.add_argument("--tgt_keep", type=float, default=0.8)
    args = parser.parse_args()
    train(args)