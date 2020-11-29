import argparse
import random
import math
import os
import sys
import time
import pprint
import datetime
import dateutil.tz

from tqdm import tqdm
import numpy as np
from PIL import Image

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.autograd import Variable, grad
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils

# from model import StyledGenerator, Discriminator

from miscc.utils import mkdir_p
from miscc.config import cfg, cfg_from_file
from datasets import TextDataset
from datasets import prepare_data
from DAMSM import RNN_ENCODER
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from styleGAN5 import StyledGenerator 
from styleGAN5 import Discriminator 
import torchvision.utils as vutils
from utils import requires_grad, accumulate, sample_data, adjust_lr



def train(args, loader, generator, discriminator, text_encoder):
    
    step = int(math.log2(args.init_size)) - 2 # fixed init_size and max_size to 64, i.e. step=4
    resolution = 4 * 2 ** step # always equal to 64
    # loader = sample_data(dataset, args.batch.get(resolution, args.batch_default), resolution)
    data_loader = iter(loader)
    adjust_lr(g_optimizer, args.lr.get(resolution, 0.001))
    adjust_lr(d_optimizer, args.lr.get(resolution, 0.001))
    requires_grad(generator, False)
    requires_grad(discriminator, True)

    disc_loss_val = 0
    gen_loss_val = 0
    grad_loss_val = 0
    alpha = 0
    used_sample = 0
    max_step = int(math.log2(args.max_size)) - 2
    final_progress = False

    for i in range(10_000_000):
        discriminator.zero_grad()
        alpha = min(1, 1 / args.phase * (used_sample + 1))
        if resolution == args.init_size or final_progress:
            alpha = 1

        # -------------------------- progressive training ------------------------
        if used_sample > args.phase * 2:
            used_sample = 0
            step += 1
            if step > max_step:
                step = max_step
                final_progress = True
            else:
                alpha = 0
            resolution = 4 * 2 ** step
            # loader = sample_data(dataset, args.batch.get(resolution, args.batch_default), resolution)
            data_loader = iter(loader)
            torch.save(
                {
                    'generator': generator.module.state_dict(),
                    'discriminator': discriminator.module.state_dict(),
                    'g_optimizer': g_optimizer.state_dict(),
                    'd_optimizer': d_optimizer.state_dict(),
                    'g_running': g_running.state_dict()
                },
                '%s/train_step-{step}.model' % args.model_path,
            )
            adjust_lr(g_optimizer, args.lr.get(resolution, 0.001))
            adjust_lr(d_optimizer, args.lr.get(resolution, 0.001))
        try:
            data = next(data_loader)
        except (OSError, StopIteration):
            data_loader = iter(loader)
            data = next(data_loader)

        imags, captions, cap_lens, class_ids, keys = prepare_data(data)
        hidden = text_encoder.init_hidden(batch_size)
        # words_embs: batch_size x nef x seq_len
        # sent_emb: batch_size x nef
        words_embs, sent_emb = text_encoder(captions, cap_lens, hidden)
        words_embs, sent_emb = words_embs.detach(), sent_emb.detach()
        real_image = imags[0]
    
        # --------------------------- start training -------------------------
        used_sample += real_image.shape[0]
        b_size = real_image.size(0)
        real_image = real_image.cuda()

        if args.loss == 'wgan-gp':
            real_predict = discriminator(real_image, sent_emb, step=step, alpha=alpha) 
            real_predict = real_predict.mean() - 0.001 * (real_predict ** 2).mean()
            (-real_predict).backward()

        elif args.loss == 'r1':
            real_image.requires_grad = True
            real_predict = discriminator(real_image, sent_emb, step=step, alpha=alpha)
            real_predict = F.softplus(-real_predict).mean()
            real_predict.backward(retain_graph=True)

            grad_real = grad(
                outputs=real_predict.sum(), inputs=real_image, create_graph=True
            )[0]
            grad_penalty = (
                grad_real.view(grad_real.size(0), -1).norm(2, dim=1) ** 2
            ).mean()
            grad_penalty = 10 / 2 * grad_penalty
            grad_penalty.backward()
            grad_loss_val = grad_penalty.item()

        if args.mixing and random.random() < 0.9: # mixing regularization might be important 
            gen_in11, gen_in12, gen_in21, gen_in22 = torch.randn(
                4, b_size, code_size, device='cuda'
            ).chunk(4, 0)
            
            gen_in11 = torch.cat((gen_in11.squeeze(0), sent_emb), 1)
            gen_in12 = torch.cat((gen_in12.squeeze(0), sent_emb), 1)
            gen_in21 = torch.cat((gen_in21.squeeze(0), sent_emb), 1)
            gen_in22 = torch.cat((gen_in22.squeeze(0), sent_emb), 1)

            gen_in1 = [gen_in11, gen_in12]
            gen_in2 = [gen_in21, gen_in22]
        else:
            gen_in1, gen_in2 = torch.randn(2, b_size, code_size, device='cuda').chunk(2, 0)
            gen_in1 = gen_in1.squeeze(0)
            gen_in2 = gen_in2.squeeze(0)
            gen_in1 = torch.cat((gen_in1, sent_emb), 1)
            gen_in2 = torch.cat((gen_in2, sent_emb), 1)

        fake_image = generator(gen_in1, step=step, alpha=alpha)
        fake_predict = discriminator(fake_image, sent_emb, step=step, alpha=alpha) 

        if args.loss == 'wgan-gp':
            fake_predict = fake_predict.mean()
            fake_predict.backward()
            eps = torch.rand(b_size, 1, 1, 1).cuda()
            x_hat = eps * real_image.data + (1 - eps) * fake_image.data
            x_hat.requires_grad = True
            hat_predict = discriminator(x_hat, sent_emb, step=step, alpha=alpha)
            grad_x_hat = grad(outputs=hat_predict.sum(), inputs=x_hat, create_graph=True)[0]
            grad_penalty = ((grad_x_hat.view(grad_x_hat.size(0), -1).norm(2, dim=1) - 1) ** 2).mean()
            grad_penalty = 10 * grad_penalty
            grad_penalty.backward()
            grad_loss_val = grad_penalty.item()
            disc_loss_val = (real_predict - fake_predict).item()

        elif args.loss == 'r1':
            fake_predict = F.softplus(fake_predict).mean()
            fake_predict.backward()
            disc_loss_val = (real_predict + fake_predict).item()

        d_optimizer.step()

        if (i + 1) % n_critic == 0:
            generator.zero_grad()
            requires_grad(generator, True)
            requires_grad(discriminator, False) 

            fake_image = generator(gen_in2, step=step, alpha=alpha) 
            predict = discriminator(fake_image, sent_emb, step=step, alpha=alpha)
            if args.loss == 'wgan-gp':
                loss = -predict.mean()
            elif args.loss == 'r1':
                loss = F.softplus(-predict).mean()
            gen_loss_val = loss.item()
            loss.backward()
            g_optimizer.step()
            accumulate(g_running, generator.module)
            requires_grad(generator, False)
            requires_grad(discriminator, True)

        # -------------------------- save intermediate results ----------------------------------
        if (i + 1) % 100 == 0:
            images = []
            gen_i, gen_j = args.gen_sample.get(resolution, (8, 8))
            with torch.no_grad():
                for _ in range(gen_i):
                    noise = torch.randn(gen_j, code_size).cuda()
                    images.append(g_running(torch.cat((noise, sent_emb[:gen_j]), 1), step=step, alpha=alpha).data.cpu())
            utils.save_image(torch.cat(images, 0), '%s/%06d.png'%(args.out_path, i), nrow=gen_i, 
                             normalize=True, range=(-1, 1),)

        if (i + 1) % 10000 == 0:
            torch.save(g_running.state_dict(), '%s/%06d.pth'%(args.model_path, i)) 

        state_msg = (f'Size: {4 * 2 ** step}; G: {gen_loss_val:.3f}; D: {disc_loss_val:.3f};'
            f' Grad: {grad_loss_val:.3f}; Alpha: {alpha:.5f}')
        print(state_msg)


if __name__ == '__main__':
    code_size = 512
    n_critic = 1

    parser = argparse.ArgumentParser(description='Progressive Growing of GANs')

    parser.add_argument('--phase', type=int, default=600_000, help='number of samples used for each training phases')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--sched', action='store_true', help='use lr scheduling')
    parser.add_argument('--init_size', default=64, type=int, help='initial image size')
    parser.add_argument('--max_size', default=64, type=int, help='max image size')
    parser.add_argument('--imsize', default=64, type=int, help='image size, no progressive training is used')
    parser.add_argument('--sentence_dim', default=256, type=int, help='sentence embedding dimension')
    parser.add_argument('--batch_size', default=32, type=int, help='batch_size')
    parser.add_argument('--mixing', action='store_true', help='use mixing regularization')
    parser.add_argument('--loss', type=str, default='wgan-gp', choices=['wgan-gp', 'r1'], help='class of gan loss')
    parser.add_argument('--eval', action='store_true', help='validation mode')
    parser.add_argument('--out_path', type=str, help='output path')
    parser.add_argument('--model_path', type=str, help='checkpoint dir')
    parser.add_argument('--cfg', type=str, default='cfg/texture.yml')
    parser.add_argument('--data_dir', type=str, default='../data/texture', help='data_path')
    parser.add_argument('--text_encoder_path', type=str, default='../DAMSMencoders/texture/text_encoder550.pth', 
                            help='pretrained text encoder path')

    args = parser.parse_args()
    cfg_from_file(args.cfg)


    # ------------------------------ dataset ------------------------------------
    imsize = args.imsize
    batch_size = args.batch_size
    image_transform = transforms.Compose([
        transforms.Resize((imsize, imsize)),
        transforms.RandomHorizontalFlip()])
    if args.eval: 
        dataset = TextDataset(args.data_dir, 'test', base_size=imsize, transform=image_transform)
        assert dataset
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, drop_last=True, 
                                                shuffle=True, num_workers=1)
    else:     
        dataset = TextDataset('../data/texture', 'train', base_size=imsize, transform=image_transform)
        print(dataset.n_words, dataset.embeddings_num)
        assert dataset
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, drop_last=True, 
                                                shuffle=True, num_workers=int(1))

    # ------------------------------ model ------------------------------------

    generator = nn.DataParallel(StyledGenerator(code_size, sentence_dim=args.sentence_dim)).cuda() 
    discriminator = nn.DataParallel(Discriminator(sentence_dim=args.sentence_dim)).cuda() 
    g_running = StyledGenerator(code_size, sentence_dim=args.sentence_dim).cuda()
    g_running.train(False)

    g_optimizer = optim.Adam(generator.module.generator.parameters(), lr=args.lr, betas=(0.0, 0.99))
    g_optimizer.add_param_group({'params': generator.module.style.parameters(), 'lr': args.lr * 0.01, 'mult': 0.01,})
    d_optimizer = optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.0, 0.99))
    accumulate(g_running, generator.module, 0)

    # test encoder
    text_encoder = RNN_ENCODER(dataset.n_words, nhidden=256)
    state_dict = torch.load(args.text_encoder_path, map_location=lambda storage, loc: storage)
    text_encoder.load_state_dict(state_dict)
    text_encoder.cuda()
    for p in text_encoder.parameters():
        p.requires_grad = False
    text_encoder.eval()    


    # ------------------------------ hyperparameters ------------------------------------
    if args.sched:
        args.lr = {128: 0.0015, 256: 0.002, 512: 0.003, 1024: 0.003}
        args.batch = {4: 512, 8: 512, 16: 512, 32: 512, 64: 128, 128: 64, 256: 32}
    else:
        args.lr = {}
        args.batch = {}
    args.gen_sample = {128:(16,16), 512: (8, 4), 1024: (4, 2), 64: (16, 16), 32: (16, 16)}
    args.batch_default = 32
    # train(args, dataset, generator, discriminator, text_encoder)
    train(args, dataloader, generator, discriminator, text_encoder)
   




