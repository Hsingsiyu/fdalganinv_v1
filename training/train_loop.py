# from datasets.celebahq import FFHQ
import datasets.celebahq
from models.stylegan_generator import StyleGANGenerator
import numpy as np
# from collections import OrderedDict

from models.stylegan_encoder import StyleGANEncoder
from models.perceptual_model import PerceptualModel
# from models.naive_discriminator import Discriminator
# from models.stylegan_discriminator import StyleGANDiscriminator
from models.stylegan_discriminator_network2 import h_layers
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
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
def mytask_loss_(x, x_rec,cuda=False):
    loss = torch.nn.L1Loss()
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
    train_dataloader = DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config.test_batch_size, shuffle=False)
    # construct model
    G = StyleGANGenerator(config.model_name, logger, gpu_ids=config.gpu_ids)
    E = StyleGANEncoder(config.model_name, logger, gpu_ids=config.gpu_ids)
    D=h_layers(config.image_size,fmaps_max=128)
     # [bn,128,h/16.w/16]

    # load parameter
    if config.gpu_ids is not None:
        assert len(config.gpu_ids) > 1
        D= nn.DataParallel(D, config.gpu_ids)

    E.net.apply(weight_init)
    D_weight='/home/xsy/idinvert_pytorch-mycode/trainStyleD_output/styleganffhq256_discriminator_epoch_199.pth'
    D_dict=D.state_dict()
    pretrained_dict=torch.load(D_weight)
    pretrained_dict ={k:v for k,v in pretrained_dict.items() if k in D_dict}
    D_dict.update(pretrained_dict)
    D.load_state_dict(D_dict)
    G.net.synthesis.eval()
    D.train()
    D=D.cuda()

    encode_dim = [G.num_layers, G.w_space_dim]

    #fDAL
    learner = fDALLearner(backbone=E.net, taskhead=D, taskloss=mytask_loss_, divergence=config.divergence,batchsize=config.train_batch_size,encoderdim=encode_dim, reg_coef=config.reg_coef, n_classes=-1,
                          grl_params={"max_iters": int((config.nepoch)*len(train_dataloader)), "hi": 0.6, "auto_step": True},Generator=G.net.synthesis,gpu_ids=config.gpu_ids # ignore for defaults.
                          )

    if config.adam:
        optimizer =torch.optim.Adam(learner.auxhead.parameters(), lr=config.learn_rate, **opt_args)
    else:
        optimizer = torch.optim.SGD(learner.auxhead.parameters(), lr=config.learn_rate, momentum=0.9,
                                    nesterov=True, weight_decay=0.02)
    optimizer_E = torch.optim.Adam(learner.backbone.parameters(), lr=E_learning_rate, **opt_args)
    lr_scheduler_E = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer_E, gamma=E_lr_args.decay_rate)

    opt_schedule=fDAL.utils.scheduler(optimizer,config.learn_rate,decay_step_=D_lr_args.decay_step,gamma_=0.5)
    global_step = 0
    for epoch in range(max_epoch):
        for step, items in enumerate(train_dataloader):
            E.net.train()
            x_s=items['x_s']
            x_t=items['x_t']
            x_s=x_s.float().cuda()
            x_t=x_t.float().cuda()

            loss,loss_val=learner((x_s,x_t),x_s)
            optimizer.zero_grad()
            optimizer_E.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(learner.backbone.parameters(), 10)
            torch.nn.utils.clip_grad_norm_(learner.auxhead.parameters(), 10)

            optimizer.step()
            optimizer_E.step()

            log_message= f"[Task Loss:(pixel){loss_val['pix_loss']:.5f}, h {loss_val['code_loss']:.8f}" \
                         f",Fdal Loss:{loss_val['fdal_loss']:.8f},src:{loss_val['fdal_src']:.8f},trg:{loss_val['fdal_trg']:.8f}] "
            # save_filename = f'epoch_{epoch:03d}_step_{step:04d}_train.png'
            # save_filepath = os.path.join(config.save_images, save_filename)
            # with torch.no_grad():
            #     tvutils.save_image(tensor=loss_val['x_all'], fp=save_filepath, nrow=config.train_batch_size, normalize=True,
            #                    scale_each=True)
            # np.savetxt(os.path.join(config.save_logs+f'h_output_epoch_{epoch:03d}_step_{step:04d}.txt'), loss_val["h_all"].cpu().detach().numpy(), fmt='%.6f', delimiter=',')
            if logger:
                logger.debug(f'Epoch:{epoch:03d}, '
                             f'Step:{step:04d}, '
                             f'lr:{optimizer.state_dict()["param_groups"][0]["lr"]:.2e}, ' # FIXME
                             f'{log_message},')
                             # f'h_output{np.array(loss_val["h_all"].cpu().detach())}')
            if writer:
                writer.add_scalar('TaskLoss/pixel', loss_val["pix_loss"].item(), global_step=global_step)
                writer.add_scalar('TaskLoss/h', loss_val["code_loss"].item(), global_step=global_step)
                writer.add_scalar('fDAL/dst', loss_val["fdal_loss"].item(), global_step=global_step)
                writer.add_scalar('fDAL/src', loss_val["fdal_src"], global_step=global_step)
                writer.add_scalar('fDAL/trg', loss_val["fdal_trg"], global_step=global_step)

            if global_step % image_snapshot_ticks == 0:
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
                    torch.save(loss_val["h_all"].cpu().detach(),
                               os.path.join(config.save_logs, f'h_output_epoch_{epoch:03d}_step_{step:04d}.pt'))

                    if val_step > config.test_save_step:
                        break
                    save_filename = f'epoch_{epoch:03d}_step_{step:04d}_test_{val_step:04d}.png'
                    save_filepath = os.path.join(config.save_images, save_filename)
                    tvutils.save_image(tensor=x_all, fp=save_filepath, nrow=batch_size_val, normalize=True,
                                       scale_each=True)

            global_step += 1
            if (global_step) % E_lr_args.decay_step == 0:
                lr_scheduler_E.step()
            opt_schedule.step()

        if epoch % 50 == 0:
            save_filename = f'styleganinv_encoder_epoch_{epoch:03d}.pth'
            save_filepath = os.path.join(config.save_models, save_filename)
            if config.gpu_ids is not None:
                torch.save(E.net.module.state_dict(), save_filepath)
                torch.save(D.module.state_dict(),os.path.join(config.save_models,f'styleganinv_dis_h_{epoch:03d}.pth'))
            else:
                torch.save(E.net.state_dict(), save_filepath)
                torch.save(D.state_dict(),os.path.join(config.save_models, f'styleganinv_dis_h_{epoch:03d}.pth'))
        # torch.save(E.net.module.state_dict(), save_filepath)  #nGPU

