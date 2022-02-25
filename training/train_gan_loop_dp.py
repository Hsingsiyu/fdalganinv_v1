# from datasets.celebahq import FFHQ
import datasets.celebahq
from models.stylegan_generator import StyleGANGenerator
import numpy as np

from models.stylegan_encoder import StyleGANEncoder
from models.stylegan_discriminator_network2 import h_layers
from training.misc import EasyDict

import os
import torch
import torch.nn as nn
import torchvision.utils as tvutils
from torch.utils.data import DataLoader

from fDAL import fDALLearner
import fDAL.utils
import  copy
import random
from fDAL.utils import ConjugateDualFunction
from collections import OrderedDict

#TODO 随机数种子怎么搞啊
def same_seeds(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
        torch.backends.cudnn.benchmark = True

def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
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
    same_seeds(2022)#
    epoch_s=0
    D_iters = config.D_iters
    E_iterations=0
    D_iterations=0
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # construct dataloader
    train_dataset=datasets.celebahq.ImageDataset(dataset_args,train=True)
    val_dataset = datasets.celebahq.ImageDataset(dataset_args, train=False)
    train_dataloader = DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config.test_batch_size, shuffle=False)
    # construct model
    G = StyleGANGenerator(config.model_name, logger, gpu_ids=config.gpu_ids)
    E = StyleGANEncoder(config.model_name, logger, gpu_ids=config.gpu_ids)
    D=h_layers(config.image_size,fmaps_max=128)
    D_hat = copy.deepcopy(D)
    D_hat.apply(lambda self_: self_.reset_parameters() if hasattr(self_, 'reset_parameters') else None)
    D.to(device)
    D_hat.to(device)
     # [bn,128,h/16.w/16]
    # load parameter
    if config.gpu_ids is not None:
        assert len(config.gpu_ids) > 1
        D= nn.DataParallel(D, config.gpu_ids)
        D_hat=nn.DataParallel(D_hat,config.gpu_ids)

    G.net.synthesis.eval()
    D.eval()

    encode_dim = [G.num_layers, G.w_space_dim]

    optimizer_E = torch.optim.Adam(E.net.parameters(), lr=E_lr_args.learning_rate, **opt_args)
    if config.adam:
        optimizer_Dhat =torch.optim.Adam(D_hat.parameters(), lr=config.learn_rate, **opt_args)
    else:
        optimizer_Dhat = torch.optim.SGD(D_hat.parameters(), lr=config.learn_rate, momentum=0.9,
                                    nesterov=True, weight_decay=0.02)

    E.net.apply(weight_init)
    D_weight='/home/xsy/idinvert_pytorch-mycode/trainStyleD_output/styleganffhq256_discriminator_epoch_199.pth'
    D_dict=D.state_dict()
    pretrained_dict=torch.load(D_weight)
    pretrained_dict ={k:v for k,v in pretrained_dict.items() if k in D_dict}
    D_dict.update(pretrained_dict)
    D.load_state_dict(D_dict)

    if config.netE!='':
        E.net.load_state_dict(torch.load(config.netE))

    if config.netD_hat!='':
        checkpoint=torch.load(config.netD_hat)
        D_hat.load_state_dict(checkpoint["h_hat"])
        optimizer_Dhat.load_state_dict(checkpoint["optD_hat"])
        optimizer_E.load_state_dict((checkpoint["optE"]))
        epoch_s=checkpoint["epoch"]

    #特别详细的参数选择的记录
    lr_scheduler_E = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer_E, gamma=E_lr_args.decay_rate)
    lr_scheduler_Dhat=fDAL.utils.scheduler(optimizer_Dhat,config.learn_rate,decay_step_=D_lr_args.decay_step,gamma_=0.5)

    l_func = nn.MSELoss(reduction='none')
    phistar_gf = lambda t: fDAL.utils.ConjugateDualFunction(config.divergence).fstarT(t)

    for epoch in range(epoch_s,max_epoch):
        data_iter = iter(train_dataloader)
        i=0
        if epoch<50:
            image_snapshot_step=50
        else:
            image_snapshot_step=image_snapshot_ticks
        while i<len(train_dataloader):
            i+=1
            data=data_iter.next()
            x_s = data['x_s']
            x_t = data['x_t']
            x_s = x_s.float().cuda(non_blocking=True)
            x_t = x_t.float().cuda(non_blocking=True)
            batch_size = x_t.shape[0]

            if (i+1)%(D_iters+1)!=0:
                ############################
                # (1) Update D' network
                ############################
                E.net.eval()
                w_s = E.net(x_s).view(batch_size, *encode_dim)
                w_t = E.net(x_t).view(batch_size, *encode_dim)

                xrec_s = G.net.synthesis(w_s)
                xrec_t = G.net.synthesis(w_t)

                features_s = D(xrec_s)  # h(GE(x_s))
                features_t = D(xrec_t)  # h(GE(x_s))

                features_s_adv = D_hat(xrec_s)  # h'(GE(x_s))
                features_t_adv = D_hat(xrec_t)  # h'(GE(x_t))

                l_s = l_func(features_s_adv, features_s)
                l_t = l_func(features_t_adv, features_t)
                dst = torch.mean(l_s) - torch.mean(phistar_gf(l_t))
                optimizer_Dhat.zero_grad()
                loss_Dhat = -dst
                loss_Dhat.backward()
                torch.nn.utils.clip_grad_norm_(D_hat.parameters(), 10)
                optimizer_Dhat.step()
                D_iterations += 1
                lr_scheduler_Dhat.step()
                if writer:
                    writer.add_scalar('trainD/dst', loss_Dhat.item(), global_step=D_iterations)
                    writer.add_scalar('trainD/src', l_s.mean().item(), global_step=D_iterations)
                    writer.add_scalar('trainD/trg', l_t.mean().item(), global_step=D_iterations)
            else:
                ############################
                # (2) Update E network
                ############################
                E.net.train()
                w_s=E.net(x_s).view(batch_size, *encode_dim)
                w_t=E.net(x_t).view(batch_size, *encode_dim)

                xrec_s = G.net.synthesis(w_s)
                xrec_t = G.net.synthesis(w_t)

                source_label = D(x_s)  # h(x_s)
                features_s = D(xrec_s)
                features_t = D(xrec_t)

                # task loss in pixel and code
                task_loss_pix = mytask_loss_(x_s, xrec_s)  # L(x_s,G(E(x_s)))
                task_loss_z = mytask_loss_(features_s, source_label)  # L(h(x),hGE(x))

                features_s_adv = D_hat(xrec_s)  # h'(GE(x_s))
                features_t_adv = D_hat(xrec_t)  # h'(GE(x_t))

                l_s = l_func(features_s_adv, features_s)
                l_t = l_func(features_t_adv, features_t)
                dst =  torch.mean(l_s) - torch.mean(phistar_gf(l_t))
                optimizer_E.zero_grad()
                loss_E=task_loss_pix+task_loss_z+dst
                loss_E.backward()
                optimizer_E.step()
                E_iterations+=1
                if writer:
                    writer.add_scalar('trainE/pixel', task_loss_pix.item(), global_step=E_iterations)
                    writer.add_scalar('trainE/h',task_loss_z.item(), global_step=E_iterations)
                    writer.add_scalar('trainE/dst', dst.item(), global_step=E_iterations)
                    writer.add_scalar('trainE/src', l_s.mean().item(), global_step=E_iterations)
                    writer.add_scalar('trainE/trg', l_t.mean().item(), global_step=E_iterations)
                log_message= f"[Task Loss:(pixel){task_loss_pix.cpu().detach().numpy():.5f}, h {task_loss_z.cpu().detach().numpy():.5f}" \
                             f", Fdal Loss:{dst.cpu().detach().numpy():.5f},src:{l_s.mean().cpu().detach().numpy():.5f},trg:{l_t.mean().cpu().detach().numpy():.5f}] "
                if logger:
                    logger.debug(f'Epoch:{epoch:03d}, '
                                 f'E_Step:{i:04d}, '
                                 f'Dlr:{optimizer_Dhat.state_dict()["param_groups"][0]["lr"]:.2e}, '
                                 f'Elr:{optimizer_Dhat.state_dict()["param_groups"][0]["lr"]:.2e}, '
                                 f'{log_message}')
                if E_iterations % image_snapshot_step == 0:
                    E.net.eval()
                    with torch.no_grad():
                        x_train = torch.cat([x_s, xrec_s, x_t, xrec_t], dim=0)
                    save_filename = f'train_epoch{epoch:03d}_step_{i:04d}.png'
                    save_filepath = os.path.join(config.save_images, save_filename)
                    tvutils.save_image(tensor=x_train, fp=save_filepath, nrow=batch_size, normalize=True,
                                       scale_each=True)
                    for val_step, val_items in enumerate(val_dataloader):
                        if val_step > config.test_save_step:
                            break
                        with torch.no_grad():
                            x_s = val_items['x_s']
                            x_t = val_items['x_t']

                            x_s = x_s.float().cuda()
                            x_t = x_t.float().cuda()

                            batch_size = x_t.shape[0]

                            w_s = E.net(x_s).view(batch_size, *encode_dim)
                            w_t = E.net(x_t).view(batch_size, *encode_dim)

                            xrec_s = G.net.synthesis(w_s)
                            xrec_t = G.net.synthesis(w_t)

                            x_all = torch.cat([x_s, xrec_s, x_t, xrec_t], dim=0)
                        save_filename = f'epoch_{epoch:03d}_step_{i:04d}_test_{val_step:04d}.png'
                        save_filepath = os.path.join(config.save_images, save_filename)
                        tvutils.save_image(tensor=x_all, fp=save_filepath, nrow=batch_size, normalize=True,
                                           scale_each=True)

                if (E_iterations+1) % E_lr_args.decay_step == 0:
                    lr_scheduler_E.step()
                    # lr_scheduler_Dhat.step()


        if (epoch+1) % 100 == 0:
            save_filename = f'styleganinv_encoder_epoch_{epoch:03d}.pth'
            save_filepath = os.path.join(config.save_models, save_filename)
            torch.save(E.net.state_dict(), save_filepath)
            checkpoint = {"h_hat": D_hat.state_dict(),
                          "optD_hat": optimizer_Dhat.state_dict(),
                          "optE": optimizer_E.state_dict(),
                          "epoch": epoch + 1}
            path_checkpoint = "{0}/checkpoint_{1}_epoch.pkl".format(config.save_models, epoch + 1)
            torch.save(obj=checkpoint, f=path_checkpoint)
