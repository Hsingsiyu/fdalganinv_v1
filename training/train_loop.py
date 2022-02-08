# from datasets.celebahq import FFHQ
import datasets.celebahq
from models.stylegan_generator import StyleGANGenerator
import numpy as np
# from collections import OrderedDict

from models.stylegan_encoder import StyleGANEncoder
from models.perceptual_model import PerceptualModel
# from models.naive_discriminator import Discriminator
# from models.stylegan_discriminator import StyleGANDiscriminator
from models.stylegan_discriminator_network import h_layer,FirstConvBlock,block
from training.misc import EasyDict

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as tvutils
import torch.autograd as autograd
from torch.utils.data import DataLoader

from fDAL import fDALLearner
import fDAL.utils
# the difference last col?

def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
    # 也可以判断是否为conv2d，使用相应的初始化方式
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
     # 是否为批归一化层
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

def mytask_loss_(x, x_rec,cuda=False):
    loss = torch.nn.MSELoss()
    error=loss(x,x_rec)
    return error

def training_loop(
        config,
        dataset_args={},
        E_lr_args=EasyDict(),
        D_lr_args=EasyDict(),
        opt_args=EasyDict(),
        logger=None,
        writer=None,
        image_snapshot_ticks=500,
        max_epoch=100
):

    E_learning_rate = E_lr_args.learning_rate

    # construct dataloader
    train_dataset=datasets.celebahq.ImageDataset(dataset_args,train=True)
    val_dataset = datasets.celebahq.ImageDataset(dataset_args, train=False)
    # num_workers???
    train_dataloader = DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config.test_batch_size, shuffle=False)
    # construct model
    G = StyleGANGenerator(config.model_name, logger, gpu_ids=config.gpu_ids)
    E = StyleGANEncoder(config.model_name, logger, gpu_ids=config.gpu_ids)

    if config.vgg:
        D = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)),
            nn.LeakyReLU(0.02),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.utils.spectral_norm(nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)),
            nn.LeakyReLU(0.02),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.utils.spectral_norm(nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False)),
            nn.LeakyReLU(0.02),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.utils.spectral_norm(nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False)),
            nn.LeakyReLU(0.02),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
    else:
        D=h_layer(config.image_size,fmaps_max=128)
     # [bn,128,h/16.w/16]

    # load parameter
    E.net.apply(weight_init)
    D_weight='/home/xsy/idinvert_pytorch-mycode/models/pretrain/styleganffhq256_discriminator_epoch_120.pth'
    D_dict=D.state_dict()
    pretrained_dict=torch.load(D_weight)
    pretrained_dict ={k:v for k,v in pretrained_dict.items() if k in D_dict}
    D_dict.update(pretrained_dict)
    D.load_state_dict(D_dict)
    G.net.synthesis.eval()
    D.train()#todo ?????
    D=D.cuda()

    encode_dim = [G.num_layers, G.w_space_dim]

    #fDAL
    learner = fDALLearner(backbone=E.net, taskhead=D, taskloss=mytask_loss_, divergence=config.divergence,batchsize=config.train_batch_size,encoderdim=encode_dim, reg_coef=config.reg_coef, n_classes=-1,
                          grl_params={"max_iters": int((config.nepoch)*len(train_dataloader)), "hi": 0.6, "auto_step": True},Generator=G.net.synthesis, # ignore for defaults.
                          )


    #TODO:  learner.paramters ????
    #optimizer=torch.optim.SGD(filter(lambda p :p.requires_grad,learner.parameters()),lr=E_learning_rate,momentum=0.9,nesterov=True,weight_decay=0.02)

    optimizer=torch.optim.SGD([learner.backbone.parameters(),
                               learner.aux_head.parameters()],lr=E_learning_rate,momentum=0.9,nesterov=True,weight_decay=0.02)


    opt_schedule=fDAL.utils.scheduler(optimizer,E_learning_rate,decay_step_=15000,gamma_=0.5)
    global_step = 0
    for epoch in range(max_epoch):
        for step, items in enumerate(train_dataloader):
            E.net.train()
            # read data
            x_s=items['x_s']
            x_t=items['x_t']
            x_s=x_s.float().cuda()
            x_t=x_t.float().cuda()

            loss,loss_val=learner((x_s,x_t))
            optimizer.zero_grad()
            loss.backward()
            # TODO  :clip grad norm?
            optimizer.step()
            log_message= f"[Task Loss:(pixel){loss_val['pix_loss']:.5f}, code {loss_val['code_loss']:.8f}" \
                         f"Fdal Loss:{loss_val['fdal_loss']:.8f}] "

            if logger:
                logger.debug(f'Epoch:{epoch:03d}, '
                             f'Step:{step:04d}, '
                             f'lr:{optimizer.state_dict()["param_groups"][0]["lr"]:.2e}, ' # FIXME
                             f'{log_message}')
            if writer:
                writer.add_scalar('TaskLoss/pixel', loss_val["pix_loss"].item(), global_step=global_step)
                writer.add_scalar('TaskLoss/h', loss_val["code_loss"].item(), global_step=global_step)
                writer.add_scalar('fDAL/dst', loss_val["fdal_loss"].item(), global_step=global_step)
                writer.add_scalar('fDAL/src', loss_val["fdal_src"], global_step=global_step)
                writer.add_scalar('fDAL/trg', loss_val["fdal_trg"], global_step=global_step)

            if step % image_snapshot_ticks == 0:
                E.net.eval()
                for val_step, val_items in enumerate(val_dataloader):
                    # FIXME
                    with torch.no_grad():
                        x_val=val_items['x_t']
                        x_val = x_val.float().cuda()
                        batch_size_val = x_val.shape[0]
                        x_train = x_s[:batch_size_val, :, :, :]
                        z_train = E.net(x_train).view(batch_size_val, *encode_dim)
                        x_rec_train = G.net.synthesis(z_train)
                        z_val = E.net(x_val).view(batch_size_val, *encode_dim)
                        x_rec_val = G.net.synthesis(z_val)
                        x_all = torch.cat([x_val, x_rec_val, x_train, x_rec_train], dim=0)
                    if val_step > config.test_save_step:
                        break
                    save_filename = f'epoch_{epoch:03d}_step_{step:04d}_test_{val_step:04d}.png'
                    save_filepath = os.path.join(config.save_images, save_filename)
                    tvutils.save_image(tensor=x_all, fp=save_filepath, nrow=batch_size_val, normalize=True,
                                       scale_each=True)

            global_step += 1
            E.net.train()
            opt_schedule.step()

        if epoch % 10 == 0:
            save_filename = f'styleganinv_encoder_epoch_{epoch:03d}'
            save_filepath = os.path.join(config.save_models, save_filename)
            torch.save(E.net.state_dict(), save_filepath)
            torch.save(D.state_dict(),os.path.join(config.save_models,f'styleganinv_dis_h_{epoch:03d}'))
        # torch.save(E.net.module.state_dict(), save_filepath)  #nGPU

