from torch.utils.data import DataLoader
from torch import optim
from torch import autograd
from torch import nn
from torchvision.utils import save_image
from torchvision import datasets, transforms
from torch.utils.data import sampler
from argparse import ArgumentParser
from wgan64x64 import *
from sklearn import metrics
import torch
import numpy as np
import time
import os
import sys
import random
import matplotlib.pyplot as plt
import lungdataset
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
sys.path.append(os.getcwd())


# Download CIFAR-10 (Python version) at
# https://www.cs.toronto.edu/~kriz/cifar.html and fill in the path to the
# extracted files here!

IMG_GAN_DIR = "/data15/boxi/anomaly detection/runs/imgsample/GAN/iter10000/"
IMG_Encoder_DIR = "/data15/boxi/anomaly detection/runs/imgsample/Encoder/iter10000/"
PTH_DIR = "/data15/boxi/anomaly detection/runs/pth_dir/iter10000/"
TB_DIR = "/data15/boxi/anomaly detection/runs/tb_dir/iter10000/" + datetime.now().strftime("%Y%m%d-%H%M%S")
EVAL_DIR = "/data15/boxi/anomaly detection/runs/eval_dir/iter10000/"

if not os.path.exists(IMG_GAN_DIR):
    os.makedirs(IMG_GAN_DIR)
if not os.path.exists(IMG_Encoder_DIR):
    os.makedirs(IMG_Encoder_DIR)
if not os.path.exists(PTH_DIR):
    os.makedirs(PTH_DIR)
if not os.path.exists(TB_DIR):
    os.makedirs(TB_DIR)
if not os.path.exists(EVAL_DIR):
    os.makedirs(EVAL_DIR)

writer = SummaryWriter(log_dir=TB_DIR)

DATAROOT = "/data15/boxi/anomaly detection/data/"

MODE = 'wgan-gp'  # Valid options are dcgan, wgan, or wgan-gp
DIM = 64  # This overfits substantially; you're probably better off with 64
LAMBDA = 10  # Gradient penalty lambda hyperparameter
CRITIC_ITERS = 5  # How many critic iterations per generator iteration
BATCH_SIZE = 64  # Batch size
ITERS = 10000  # How many generator iterations to train for
OUTPUT_DIM = 3 * 64 * 64  # Number of pixels in CIFAR10 (3*32*32)

# ---- for dataset
VALIDATION_SPLIT = 0.1
NOISE_SIZE = 128
RANDOM_SEED = 42
SHUFFLE_DB = True
# ----

torch.manual_seed(0)
np.random.seed(0)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.determinstic = False


def calc_gradient_penalty(netD, real_data, fake_data):
    alpha = torch.rand(BATCH_SIZE, 1)
    alpha = alpha.expand(BATCH_SIZE, int(
        real_data.nelement()/BATCH_SIZE)).contiguous()
    alpha = alpha.view(BATCH_SIZE, 3, DIM, DIM)
    alpha = alpha.to(device)

    fake_data = fake_data.view(BATCH_SIZE, 3, DIM, DIM)
    interpolates = alpha * real_data.detach() + ((1 - alpha) * fake_data.detach())

    interpolates = interpolates.to(device)
    interpolates.requires_grad_(True)

    disc_interpolates = netD(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(
                                  disc_interpolates.size()).to(device),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty


def one_class_dataloader(c, nw, bs):
    """
    get one class data batch
    :param c: label
    :param nw: num_workers
    :param bs: batch_size
    :return: trainloader, valloader
    """
    transform = transforms.Compose([
        # transforms.RandomCrop(32, padding=4),
        # transforms.RandomHorizontalFlip(),
        transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5),
                             (0.5, 0.5, 0.5)),
    ])
    train_db = lungdataset.PneumoniaDataset(root=DATAROOT, train=True, transform=transform)

    # Train loader
    normal_indices = train_db.target_img_dict[0]
    trainloader = DataLoader(
        train_db, bs, sampler=sampler.SubsetRandomSampler(normal_indices),
        num_workers=nw, pin_memory=True, drop_last=True)

    # Val loader
    db_size = len(train_db)
    indices = list(range(db_size))
    split = int(np.floor(VALIDATION_SPLIT * db_size))
    if SHUFFLE_DB:
        np.random.seed(RANDOM_SEED)
        np.random.shuffle(indices)

    val_indices = indices[:split]
    valloader = DataLoader(
        train_db, bs * 2, num_workers=nw, pin_memory=True,
        sampler=sampler.SubsetRandomSampler(val_indices))

    # Test loader
    test_db = lungdataset.PneumoniaDataset(root=DATAROOT, train=False, transform=transform)

    return trainloader, valloader

def wgan_training():
    netG = GoodGenerator().to(device)
    netD = GoodDiscriminator().to(device)

    one = torch.FloatTensor([1]).to(device)
    mone = one * -1

    # TODO
    # 不要用基于动量的优化算法（Adam，momentum）
    # 建议改为RMSProp, SGD
    # optimizerD = optim.Adam(netD.parameters(), lr=1e-4, betas=(0.0, 0.9))
    # optimizerG = optim.Adam(netG.parameters(), lr=1e-4, betas=(0.0, 0.9))

    optimizerD = optim.RMSprop(netD.parameters(), lr=1e-4)
    optimizerG = optim.RMSprop(netG.parameters(), lr=1e-4)

    dataloader, _ = one_class_dataloader(options.c, 2, BATCH_SIZE)
    D_real_list, D_fake_list, D_cost_list, G_cost_list = [], [], [], []

    for iteration in range(ITERS):
        start_time = time.time()
        ############################
        # (1) Update D network
        # lossD ↓ = D(fake data) sum mean - D(real data) sum mean + gradient penalty
        # Wasserstein distance: D(real data) sum mean - D(fake data) sum mean
        # represent distance between real data and fake data
        # smaller Wassertein distance, better model
        ###########################
        for i, (_data, target, _) in enumerate(dataloader):
            # _data : [64,3,64,64]
            if i == CRITIC_ITERS:
                break
            netD.zero_grad()

            # train with real

            real_data = _data.to(device)

            D_real = netD(real_data)
            D_real = D_real.mean()
            writer.add_scalar('Score/real-a-batch', D_real, iteration * CRITIC_ITERS + i)

            D_real.backward(mone)
            D_real_list.append(D_real.item())

            # train with fake
            noise = torch.randn(BATCH_SIZE, NOISE_SIZE)
            noise = noise.to(device)
            fake = netG(noise).detach()
            inputv = fake
            D_fake = netD(inputv)
            D_fake = D_fake.mean()
            writer.add_scalar('Score/fake-a-batch', D_fake, iteration * CRITIC_ITERS + i)

            D_fake.backward(one)
            D_fake_list.append(D_fake.item())

            # train with gradient penalty
            gradient_penalty = calc_gradient_penalty(
                netD, real_data.data, fake.data)
            gradient_penalty.backward()

            # backward()函数会自动把导数值叠加，除非调用zero_grad清空
            # print "gradien_penalty: ", gradient_penalty

            D_cost = D_fake - D_real + gradient_penalty
            writer.add_scalar('Loss/DLoss', D_cost, iteration * CRITIC_ITERS + i)
            D_cost_list.append(D_cost.item())

            Wasserstein_D = D_real - D_fake
            writer.add_scalar('Wassertein D', Wasserstein_D, iteration * CRITIC_ITERS + i)

            optimizerD.step()
        ############################
        # (2) Update G network
        # Loss G = - D(fake data) sum mean
        # ↓ Loss G, ↑ D(fake data), means Generator cheat Discriminator
        ###########################
        netG.zero_grad()

        noise = torch.randn(BATCH_SIZE, 128)
        noise = noise.to(device)
        fake = netG(noise)
        G = netD(fake)
        G = G.mean()
        G.backward(mone)  # 不会改变G的值，但会把负号加到bp过程中

        G_cost = -G
        writer.add_scalar('Loss/GLoss', G_cost, iteration)

        optimizerG.step()
        G_cost_list.append(G_cost.item())

        # Write logs and save samples
        if iteration % 20 == 0:
            print('Iters:{}, D(real):{}, D(fake):{}, Loss D:{}, Loss G:{}'.format(
                iteration,
                np.mean(D_real_list),
                np.mean(D_fake_list),
                np.mean(D_cost_list),
                np.mean(G_cost_list),)
            )
        if iteration % 1000 == 0 and iteration != 0:
            save_image(fake * 0.5 + 0.5, IMG_GAN_DIR + '{}.jpg'.format(iteration))
            torch.save(netD.state_dict(), PTH_DIR + 'netD_%d.pth' % iteration)
            torch.save(netG.state_dict(), PTH_DIR + 'netG_%d.pth' % iteration)


def train_encoder():
    netG = GoodGenerator().to(device)
    netG.load_state_dict(torch.load(PTH_DIR + 'netG_9000.pth'))
    netG.eval()
    netD = GoodDiscriminator().to(device)
    netD.load_state_dict(torch.load(PTH_DIR + 'netD_9000.pth'))
    netD.eval()
    for p in netD.parameters():
        p.requires_grad = False
    for p in netG.parameters():
        p.requires_grad = False

    dataloader, _ = one_class_dataloader(options.c, 2, BATCH_SIZE)

    netE = Encoder(DIM, NOISE_SIZE).to(device)
    # netE.load_state_dict(torch.load(PTH_DIR + 'netE.pth'))

    optimizer = optim.Adam(netE.parameters(), 1e-4, (0.0, 0.9))

    crit = nn.MSELoss()

    for e in range(300):
        losses = []
        netE.train()
        for (x, _, _) in dataloader:
            x = x.to(device)
            code = netE(x)
            rec_image = netG(code)
            d_input = torch.cat((x, rec_image), dim=0)
            f_x, f_gx = netD.extract_feature(d_input).chunk(2, 0)
            loss = crit(rec_image, x) + crit(f_gx, f_x.detach())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        print(e, np.mean(losses))
        netE.eval()
        rec_image = netG(netE(x))
        d_input = torch.cat((x, rec_image), dim=0)
        save_image(d_input*0.5+0.5, IMG_Encoder_DIR + 'rec'+str(e)+'.bmp')
    torch.save(netE.state_dict(), PTH_DIR + 'netE.pth')


def evaluate():
    """
    in_real: 所选定class的图片
    in_fake: 所选定class的生成图片
    out_real: 非选定class的图片
    out_fake: 非选定class的生成图片
    rec_diff: reconstruct difference. 真实图片和生成图片间的像素级距离（欧几里得距离）
    rec_score: 根据rec_diff获得
    f_x, f_gx: discriminator所抽取出来的，真实图片特征与生成图片特征。
    feat_diff：由f_x和f_gx得到的距离。值越小，表示generator越能好的生成图片
    feat_score: 有feat_diff得来
    outlier_score: a * rec_score + b * feat_score。值越大，越差。
    y_true: 所有图片的label。为所选定class的为1，非选定class的为-1
    y_score: discriminator。给予每张图片的分数。分数有outlier_score得来。
    auc：计算y_true与-y_score的ROC

    :return:
    """
    netG = GoodGenerator().to(device)
    netG.load_state_dict(torch.load(PTH_DIR + 'netG_9000.pth'))
    netG.eval()
    netD = GoodDiscriminator().to(device)
    netD.load_state_dict(torch.load(PTH_DIR + 'netD_9000.pth'))
    netD.eval()
    netE = Encoder(DIM, NOISE_SIZE).to(device)
    netE.load_state_dict(torch.load(PTH_DIR + 'netE.pth'))
    netE.eval()

    _, dataloader = one_class_dataloader(options.c, 0, BATCH_SIZE)
    # crit = nn.MSELoss()
    y_true, y_score = [], []
    in_real, out_real, in_fake, out_fake = [], [], [], []
    with torch.no_grad():
        for (x, label, _) in dataloader:
            bs = x.size(0)
            x = x.to(device)
            fake_image = netG(netE(x))
            d_input = torch.cat((x, fake_image), dim=0)
            idx = (label == options.c)
            in_real.append(x[idx])
            in_fake.append(fake_image[idx])
            idx = (label != options.c)
            out_real.append(x[idx])
            out_fake.append(fake_image[idx])
            f_x, f_gx = netD.extract_feature(d_input).chunk(2, 0)
            a, b = 1, 1
            rec_diff = ((fake_image.view(bs, -1) - x.view(bs, -1))**2)
            rec_score = rec_diff.mean(dim=1) - rec_diff.std(dim=1)
            feat_diff = ((f_x - f_gx)**2)
            feat_score = feat_diff.mean(dim=1) + feat_diff.std(dim=1)
            outlier_score = a * rec_score + b * feat_score
            y_true.append(label)
            y_score.append(outlier_score.cpu())
    in_real = torch.cat(in_real, dim=0)[:32]
    in_fake = torch.cat(in_fake, dim=0)[:32]
    out_real = torch.cat(out_real, dim=0)[:32]
    out_fake = torch.cat(out_fake, dim=0)[:32]

    save_image(torch.cat((in_real, in_fake), dim=0), EVAL_DIR + 'real.bmp', normalize=True)
    save_image(torch.cat((out_real, out_fake), dim=0), EVAL_DIR + 'fake.bmp', normalize=True)
    y_score = np.concatenate(y_score)
    y_true = np.concatenate(y_true)
    y_true[y_true != options.c] = -1
    y_true[y_true == options.c] = 1
    print('auc:', metrics.roc_auc_score(y_true, -y_score))


if __name__ == '__main__':
    parser = ArgumentParser()
    '''alpha: scales the lr of Discriminator'''
    '''beta: controls the trade-off between rec loss and prior loss'''
    '''gamma: controls the trade-off between rec loss and gen loss'''
    parser.add_argument('--alpha', dest='a', type=float, default=0.1)
    parser.add_argument('--beta', dest='b', type=float, default=5)
    parser.add_argument('--gamma', dest='g', type=float, default=15)
    parser.add_argument('--eval', dest='eval', action='store_true')
    parser.add_argument('--class', dest='c', type=int, required=True)
    parser.add_argument('--cuda', dest='cuda', type=str, required=True)
    global options
    options = parser.parse_args()
    device = torch.device('cuda:{}'.format(options.cuda))
    torch.cuda.set_device('cuda:{}'.format(options.cuda))
    if not options.eval:
        # wgan_training()
        train_encoder()
    else:
        evaluate()
        # find_match_img()

    writer.close()
