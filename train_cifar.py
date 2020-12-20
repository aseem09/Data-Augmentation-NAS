import os
import sys
import time
import glob
import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
import genotypes
import torch.utils
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn

from torch.autograd import Variable
from model import NetworkCIFAR as Network
from torchvision.utils import save_image
from generator import Generator
from classifier_model import Resnet18
from genotypes import PRIMITIVES
from genotypes import Genotype

import copy
import torchvision.utils as vutils
from torchvision.utils import save_image

parser = argparse.ArgumentParser("cifar")
parser.add_argument('--workers', type=int, default=4, help='number of workers')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--epochs', type=int, default=600, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=36, help='num of init channels')
parser.add_argument('--layers', type=int, default=20, help='total number of layers')
parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
parser.add_argument('--cutout', action='store_true', default=True, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--save', type=str, default='', help='experiment name')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--arch', type=str, default='PDARTS', help='which architecture to use')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--tmp_data_dir', type=str, default='/tmp/cache/', help='temp data dir')
parser.add_argument('--note', type=str, default='try', help='note for this run')
parser.add_argument('--cifar100', action='store_true', default=False, help='if use cifar100')

parser.add_argument('--latent_dim', type=int, default=100, help='size of noise vector')
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")

args, unparsed = parser.parse_known_args()

args.save= '/ceph/aseem-volume/datagan/search/2/logging'
args.tmp_data_dir= '/ceph/aseem-volume/datagan/search/2/data'
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

if args.cifar100:
    CIFAR_CLASSES = 100
    data_folder = 'cifar-100-python'
else:
    CIFAR_CLASSES = 10
    data_folder = 'cifar-10-batches-py'

cuda = True if torch.cuda.is_available() else False

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

FT = torch.cuda.LongTensor
FT_a = torch.cuda.FloatTensor


# Loss functions 
a_loss = torch.nn.BCELoss()
a_loss.cuda()

# Labels 
real_label = 0.9
fake_label = 0.0

gamma = 0.01
lambda_gp = 10

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def sample_image(gen, batches_done, n_row=10):
    """Saves a grid of generated digits ranging from 0 to n_classes"""
    # Sample noise
    z = Variable(FloatTensor(np.random.normal(0, 1, (n_row ** 2, args.latent_dim))))
    # Get labels ranging from 0 to n_classes for n rows
    labels = np.array([num for _ in range(n_row) for num in range(n_row)])
    labels = Variable(LongTensor(labels))
    gen_imgs = gen(z, labels)
    save_image(gen_imgs.data, "/ceph/aseem-volume/datagan/search/2/%d.png" % batches_done, nrow=n_row, normalize=True)

def main():
    if not torch.cuda.is_available():
        logging.info('No GPU device available')
        sys.exit(1)
    np.random.seed(args.seed)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled=True
    torch.cuda.manual_seed(args.seed)
    logging.info("args = %s", args)
    logging.info("unparsed args = %s", unparsed)
    num_gpus = torch.cuda.device_count()
    
    genotype = eval("genotypes.%s" % args.arch)
    print('---------Genotype---------')
    logging.info(genotype)
    print('--------------------------')
    disc = Network(args.init_channels, CIFAR_CLASSES, args.layers, args.auxiliary, genotype)
    disc = torch.nn.DataParallel(disc)
    disc = disc.cuda()
    logging.info("param size = %fMB", utils.count_parameters_in_MB(disc))


    adversarial_loss = nn.MSELoss()
    adversarial_loss.cuda()

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    optimizer_disc = torch.optim.SGD(
        disc.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay
        )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_disc, float(args.epochs))

    gen = Generator(100)
    gen.cuda()

    model = Resnet18()
    model.cuda()

    logging.info("param size gen= %fMB", utils.count_parameters_in_MB(gen))
    logging.info("param size model= %fMB", utils.count_parameters_in_MB(model))

    optimizer_gen = torch.optim.Adam(gen.parameters(), lr=args.lr, 
                        betas=(args.b1, args.b2))

    optimizer_model = torch.optim.SGD(model.parameters(), lr=0.1,
                        momentum=0.9, weight_decay=5e-4)
    scheduler_model = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_model, T_max=200)

    if args.cifar100:
        train_transform, valid_transform = utils._data_transforms_cifar100(args)
    else:
        train_transform, valid_transform = utils._data_transforms_cifar10(args)
    if args.cifar100:
        train_data = dset.CIFAR100(root=args.tmp_data_dir, train=True, download=True, transform=train_transform)
        valid_data = dset.CIFAR100(root=args.tmp_data_dir, train=False, download=True, transform=valid_transform)
    else:
        train_data = dset.CIFAR10(root=args.tmp_data_dir, train=True, download=True, transform=train_transform)
        valid_data = dset.CIFAR10(root=args.tmp_data_dir, train=False, download=True, transform=valid_transform)

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=args.workers)

    valid_queue = torch.utils.data.DataLoader(
        valid_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.workers)
    
    best_acc = 0.0
    for epoch in range(args.epochs):
        
        # scheduler_model.step()
        # lr_gen = args.lr
        # lr_disc = args.learning_rate
        # lr_model = scheduler_model.get_lr()[0]

        # logging.info('Epoch: %d lr_model %e', epoch, lr_model)
        # disc.module.drop_path_prob = args.drop_path_prob * epoch / args.epochs
        # disc.drop_path_prob = args.drop_path_prob * epoch / args.epochs
        # start_time = time.time()
        # train_acc, train_obj = train(train_queue, gen ,disc, model, criterion, adversarial_loss, optimizer_disc, optimizer_gen, optimizer_model)
        # logging.info('Train_acc: %f', train_acc)

        # valid_acc, valid_obj = infer(valid_queue, model, criterion)
        # if valid_acc > best_acc:
        #     best_acc = valid_acc
        # logging.info('Valid_acc: %f', valid_acc)

        logging.info('Epoch: %d', epoch)
        epoch_start = time.time()
        train_acc, train_obj = train_gan(epoch, train_queue, valid_queue, gen, disc, criterion, adversarial_loss, optimizer_gen, optimizer_disc, 0, 0, 0, 0, train_arch=True)
        epoch_duration = time.time() - epoch_start
        logging.info('Epoch time: %ds', epoch_duration)

        end_time = time.time()
        duration = end_time - start_time
        print('Epoch time: %ds.' % duration )
        # utils.save(disc.module, os.path.join(args.save, 'weights.pt'))

def compute_gradient_penalty(D, real_samples, fake_samples, labels):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = torch.Tensor(np.random.random((real_samples.size(0), 1, 1, 1))).cuda()
    labels = LongTensor(labels).cuda()
    
    real_samples.cuda()
    fake_samples.cuda()
    alpha.cuda()
    

    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    interpolates.cuda()
    d_interpolates = D(interpolates, labels)
    d_interpolates.cuda()
    fake = Variable(torch.Tensor(real_samples.shape[0], 1).cuda().fill_(1.0), requires_grad=False)
    fake.cuda()

    # Get random int
    # Get gradient w.r.t. interpolates
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients.cuda()
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

def train(train_queue, gen , disc, model, criterion, adversarial_loss, optimizer_disc, optimizer_gen, optimizer_model):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()

    model.train()
    gen.train()
    disc.train()

    for step, (input, target) in enumerate(train_queue):
        input = input.cuda(non_blocking=True)
        n = input.size(0)
        batch_size = input.shape[0]
        target = target.cuda(non_blocking=True)
    
        # Adversarial ground truths
        # valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
        # fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)

        # Configure input
        real_imgs = Variable(input.type(FloatTensor))
        labels = Variable(target.type(LongTensor))

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_gen.zero_grad()

        # Sample noise and labels as generator input
        z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, args.latent_dim))))
        # gen_labels = Variable(LongTensor(np.random.randint(0, CIFAR_CLASSES, batch_size)))

        # Generate a batch of images
        gen_imgs = gen(z, labels)

        # Loss measures generator's ability to fool the discriminator
        validity = disc(gen_imgs, labels)
        g_loss = -torch.mean(validity)

        g_loss.backward()
        nn.utils.clip_grad_norm_(gen.parameters(), args.grad_clip)
        optimizer_gen.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------
        
        optimizer_disc.zero_grad()

        # Loss for real images
        validity_real = disc(real_imgs, labels)
        # d_real_loss = adversarial_loss(validity_real, valid)

        # Loss for fake images
        validity_fake = disc(gen_imgs.detach(), labels)
        # d_fake_loss = adversarial_loss(validity_fake, fake)

        gradient_penalty = compute_gradient_penalty(disc, real_imgs.data, gen_imgs.data, labels.data)
        # Total discriminator loss
        d_loss = -torch.mean(validity_real) + torch.mean(validity_fake) + lambda_gp * gradient_penalty

        d_loss.backward()
            
        nn.utils.clip_grad_norm_(disc.parameters(), args.grad_clip)
        optimizer_disc.step()

        optimizer_gen.zero_grad()

        # ---------------------
        #  Train Model
        # ---------------------
    
        # optimizer_model.zero_grad()

        # logits_1 = model(input)
        # m_loss_1 = criterion(logits_1, target)

        # # Sample noise and labels as generator input
        # z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, args.latent_dim))))
        # gen_labels = Variable(LongTensor(np.random.randint(0, CIFAR_CLASSES, batch_size)))

        # # Generate a batch of images
        # gen_imgs = gen(z, gen_labels)

        # logits_2 = model(gen_imgs)
        # m_loss_2 = gamma * criterion(logits_2, gen_labels)

        # m_loss = m_loss_1 + m_loss_2

        # m_loss.backward()

        # optimizer_model.step()

        # logits = model(input)
        # loss = criterion(logits, target)

        # prec1, prec5 = utils.accuracy(logits, target, topk=(1, 1))
        # objs.update(loss.data.item(), n)
        # top1.update(prec1.data.item(), n)
        # top5.update(prec5.data.item(), n)

        if step % args.report_freq == 0:
            sample_image(gen, step, 10)
            logging.info('Train Step: %03d G_loss: %f D_loss: %f', step, g_loss, d_loss)
            # logging.info('Train Step: %03d Acc: %f G_loss: %f D_loss: %f M_loss: %f', step, top1.avg, g_loss, d_loss, m_loss)

    # return top1.avg, objs.avg
    return 0, 0

def train_gan(epoch, train_queue, valid_queue, gen, disc, criterion, adversarial_loss, optimizer_gen, optimizer_disc, lr, lr_model, lr_gen, lr_disc, train_arch=True):
    
    for step, (input, target) in enumerate(train_queue):
        batch_size = input.shape[0]

        input = input.cuda()
        target = target.cuda(non_blocking=True)
        
        # convert img, labels into proper form 
        imgs = Variable(input.type(FT_a))
        labels = Variable(target.type(FT))

        # creating real and fake tensors of labels 
        reall = Variable(FT_a(batch_size,1).fill_(real_label))
        f_label = Variable(FT_a(batch_size,1).fill_(fake_label))

        # initializing gradient
        optimizer_gen.zero_grad() 
        optimizer_disc.zero_grad()

        #### TRAINING GENERATOR ####
        # Feeding generator noise and labels 
        noise = Variable(FT_a(np.random.normal(0, 1,(batch_size, 100))))
        gen_labels = Variable(FT(np.random.randint(0, 10, batch_size)))
        
        gen_imgs = gen(noise, gen_labels)
        
        # Ability for discriminator to discern the real v generated images 
        validity = disc(gen_imgs, gen_labels)
        
        # Generative loss function 
        # g_loss = a_loss(validity, reall)
        g_loss = -torch.mean(validity)

        # Gradients 
        g_loss.backward()
        optimizer_gen.step()


        #### TRAINING DISCRIMINTOR ####

        optimizer_disc.zero_grad()

        # Loss for real images and labels 
        validity_real = disc(imgs, labels)
        # d_real_loss = a_loss(validity_real, reall)
        d_real_loss = -torch.mean(validity_real)

        # Loss for fake images and labels 
        validity_fake = disc(gen_imgs.detach(), gen_labels)
        d_fake_loss = torch.mean(validity_fake)

        print("Real")
        print(validity_real)
        print("Fake")
        print(validity_fake)
        
        gradient_penalty = compute_gradient_penalty(disc, imgs.data, gen_imgs.data, labels.data)
        # Total discriminator loss 
        d_loss = (d_fake_loss+d_real_loss) + 10 * gradient_penalty

        # calculates discriminator gradients
        d_loss.backward()
        optimizer_disc.step()


        # disc.train()
        # n = input.size(0)
        # batch_size = input.shape[0]
        # input = input.cuda()
        # target = target.cuda(non_blocking=True)
        
        # # Configure input
        # real_imgs = Variable(input.type(FloatTensor))
        # labels = Variable(target.type(LongTensor))

        # # -----------------
        # #  Train Generator
        # # -----------------

        # optimizer_gen.zero_grad()

        # # Sample noise and labels as generator input
        # z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, args.latent_dim))))
        # # gen_labels = Variable(LongTensor(np.random.randint(0, CIFAR_CLASSES, batch_size)))

        # # Generate a batch of images
        # gen_imgs = gen(z, labels)

        # # Loss measures generator's ability to fool the discriminator
        # validity = disc(gen_imgs, labels)
        # g_loss = -torch.mean(validity)

        # g_loss.backward()
        # nn.utils.clip_grad_norm_(gen.parameters(), args.grad_clip)
        # optimizer_gen.step()

        # # ---------------------
        # #  Train Discriminator
        # # ---------------------
        
        # optimizer_disc.zero_grad()

        # # Loss for real images
        # validity_real = disc(real_imgs, labels)
        # # d_real_loss = adversarial_loss(validity_real, valid)

        # # Loss for fake images
        # validity_fake = disc(gen_imgs.detach(), labels)
        # # d_fake_loss = adversarial_loss(validity_fake, fake)

        # # Total discriminator loss
        # gradient_penalty = compute_gradient_penalty(disc, real_imgs.data, gen_imgs.data, labels.data)
        # # Total discriminator loss
        # d_loss = -torch.mean(validity_real) + torch.mean(validity_fake) + lambda_gp * gradient_penalty

        # d_loss.backward()
            
        # nn.utils.clip_grad_norm_(network_params, args.grad_clip)
        # optimizer_disc.step()

        # optimizer_gen.zero_grad()

        if step % args.report_freq == 0:
            # sample_image(gen, step, 10)
            vutils.save_image(gen_imgs, '/ceph/aseem-volume/datagan/search/2/%s.png' % step, normalize=True)
            fake = gen(noise, gen_labels)
            vutils.save_image(fake.detach(), '/ceph/aseem-volume/datagan/search/2/%s_%03d.png' % (step, epoch), normalize=True)
            logging.info('TRAIN Step: %03d Gen_loss: %f Disc_loss: %f', step, g_loss, d_loss)

    return 0, 0

def infer(valid_queue, model, criterion):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    model.eval()

    for step, (input, target) in enumerate(valid_queue):
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        with torch.no_grad():
            logits, _ = model(input)
            loss = criterion(logits, target)

        prec1, _ = utils.accuracy(logits, target, topk=(1,5))
        n = input.size(0)
        objs.update(loss.data.item(), n)
        top1.update(prec1.data.item(), n)

        if step % args.report_freq == 0:
            logging.info('Valid Step: %03d Objs: %e Acc: %f', step, objs.avg, top1.avg)

    return top1.avg, objs.avg


if __name__ == '__main__':
    start_time = time.time()
    main() 
    end_time = time.time()
    duration = end_time - start_time
    logging.info('Eval time: %ds.', duration)
    
