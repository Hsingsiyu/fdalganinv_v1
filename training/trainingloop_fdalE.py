import datasets.celebahq
from models.stylegan_generator import StyleGANGenerator
from models.perceptual_model import PerceptualModel
from models.e2style_encoders_network import BackboneEncoderFirstStage
from models.stylegan_discriminator_network2 import StyleGANDiscriminator
from training.misc import EasyDict

import os
import torch
import torch.nn as nn
import torchvision.utils as tvutils
from torch.utils.data import DataLoader
import copy
import json
from fDAL.utils import ConjugateDualFunction
from torch.nn.parallel import DistributedDataParallel as DDP
import random
from utils.lpips.lpips import LPIPS

import torch.nn.functional as F
import torch.autograd as autograd

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
    loss=nn.L1Loss()
    error=loss(x,x_rec)
    return error
def GAN_loss(scores_out, real=True):
    if real:
        return torch.mean(F.softplus(-scores_out))
    else:
        return torch.mean(F.softplus(scores_out))
def div_loss_(D, real_x, fake_x, p=2):

    x_ = real_x.requires_grad_(True)
    y_ = D(x_)
    # cal f'(x)
    grad = autograd.grad(
        outputs=y_,
        inputs=x_,
        grad_outputs=torch.ones_like(y_),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    # grad = grad.view(x_.shape[0], -1)
    # div = (grad.norm(2, dim=1) ** p).mean()
    div = (grad * grad).sum(dim=1, keepdim=True).sum(dim=2, keepdim=True).sum(dim=3, keepdim=True)
    div = torch.mean(div)
    return div
def training_loop(
        config,
        dataset_args={},
        E_lr_args=EasyDict(),
        D_lr_args=EasyDict(),
        Ehat_lr_args=EasyDict(),
        opt_args=EasyDict(),
        loss_args= EasyDict(),
        logger=None,
        writer=None,
        image_snapshot_ticks=1000,
        max_epoch=100,
        epoch_s=0,
):
    same_seeds(2022+config.local_rank)
    epoch_s=0
    E_iterations=0
    Ehat_iterations=0

    loss_pix_weight=loss_args.loss_pix_weight
    # loss_w_weight=loss_args.loss_w_weight
    loss_dst_weight=loss_args.loss_dst_weight
    loss_feat_weight=loss_args.loss_feat_weight
    loss_lpi_weight=loss_args.loss_lpi_weight
    loss_adv_weight=loss_args.loss_adv_weight#todo
    if config.gpu_ids is not None:
        torch.distributed.init_process_group(backend='nccl',)  # choose nccl as backend using gpus
        torch.cuda.set_device(config.local_rank)

    # construct dataloader
    train_dataset=datasets.celebahq.ImageDataset(dataset_args,train=True)
    val_dataset = datasets.celebahq.ImageDataset(dataset_args, train=False)
    if config.gpu_ids is not None:
        train_sampler=torch.utils.data.distributed.DistributedSampler(train_dataset)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=int(config.train_batch_size/len(config.gpu_ids)),sampler=train_sampler,pin_memory=True,drop_last=True)

        val_sampler=torch.utils.data.distributed.DistributedSampler(val_dataset)
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=int(config.test_batch_size/len(config.gpu_ids)),sampler=val_sampler,pin_memory=True,drop_last=True)
    else:
        train_dataloader = DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=config.test_batch_size, shuffle=False)

    l_func = nn.L1Loss(reduction='none')
    phistar_gf = lambda t: ConjugateDualFunction(config.divergence).fstarT(t)
    if loss_lpi_weight > 0:
        Lpips_loss = LPIPS(net_type='alex').cuda().eval()
    # construct model
    G = StyleGANGenerator(config.model_name, logger, gpu_ids=config.gpu_ids,local_rank=config.local_rank)
    F = PerceptualModel(min_val=-1, max_val=1, gpu_ids=config.gpu_ids,local_rank=config.local_rank)
    D=StyleGANDiscriminator(config.image_size,fmaps_max=128,minibatch_std_group_size=1)
    E = BackboneEncoderFirstStage(num_layers=50,mode='ir_se') #todo 可能需要调整参数  要有记录，其实n1 n2 n3 也可以改
    E_hat=copy.deepcopy(E)

    D.cuda().train()
    E.cuda().train()
    E_hat.cuda().train()

    F.net.eval()
    G.net.synthesis.eval()

    if config.gpu_ids is not None:
        assert len(config.gpu_ids) > 1
        D = DDP(D, device_ids=[config.local_rank], broadcast_buffers=False, find_unused_parameters=True)
        E_hat =DDP(E_hat, device_ids=[config.local_rank],broadcast_buffers=False, find_unused_parameters=True)
        E=DDP(E, device_ids=[config.local_rank],broadcast_buffers=False, find_unused_parameters=True)
        if loss_lpi_weight > 0:
            Lpips_loss=DDP(Lpips_loss,device_ids=[config.local_rank],broadcast_buffers=False, find_unused_parameters=True)

    # load parameter
    E.apply(weight_init)
    D.apply(weight_init)
    E_hat.apply(lambda self_: self_.reset_parameters() if hasattr(self_, 'reset_parameters') else None)

    # encode_dim = [G.num_layers, G.w_space_dim]

    # setup optimizer
    optimizer_E = torch.optim.Adam(E.parameters(), lr=E_lr_args.learning_rate, **opt_args)
    optimizer_Ehat = torch.optim.Adam(E_hat.parameters(), lr=Ehat_lr_args.learning_rate, **opt_args)
    optimizer_D = torch.optim.Adam(D.parameters(), lr=D_lr_args.learning_rate, **opt_args)

    #TODO delete config.adam

    if config.netE!='':
        with torch.no_grad():
            E.load_state_dict(torch.load(config.netE,map_location=torch.device("cpu")))

    if config.netD_hat!='':
        with torch.no_grad():
            checkpoint=torch.load(config.netD_hat,map_location=torch.device('cpu'))
        optimizer_Ehat.load_state_dict(checkpoint["optE_hat"])
        optimizer_E.load_state_dict((checkpoint["optE"]))
        D.load_state_dict(checkpoint["D"])
        E_hat.load_state_dict(checkpoint["E_hat"])
        epoch_s=checkpoint["epoch"]

    #特别详细的参数选择的记录
    lr_scheduler_E = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer_E, gamma=E_lr_args.decay_rate)
    lr_scheduler_Ehat=torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer_Ehat, gamma=Ehat_lr_args.decay_rate)
    lr_scheduler_D = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer_D, gamma=D_lr_args.decay_rate)

    if config.netD_hat!='':
        lr_scheduler_E.load_state_dict(checkpoint['schedulerE'])
        lr_scheduler_D.load_state_dict(checkpoint['schedulerD'])
        lr_scheduler_Ehat.load_state_dict(checkpoint['schedulerEhat'])

    if config.local_rank==0:
        generator_config = {"imageSize": config.image_size, "dataset": config.dataset_name,"split":dataset_args.split ,"Dhat_iters":config.D_iters,
                            "train_bs": config.train_batch_size,"val_bs": config.test_batch_size,"div":config.divergence,"nepoch":config.nepoch,
                            "lr_E":optimizer_E.state_dict()["param_groups"][0]["lr"],
                            "lr_Dhat":optimizer_Ehat.state_dict()["param_groups"][0]["lr"],
                            "pix_weight":loss_args.loss_pix_weight,"w_weight":loss_args.loss_w_weight,"dst_weight":loss_args.loss_dst_weight,
                            "adv_weight":loss_args.loss_adv_weight,
                            "lpips_weight":loss_args.loss_lpi_weight,
                            "Edecay_rate":E_lr_args.decay_step, "Ehat_decay_rate":Ehat_lr_args.decay_step,
                            }
        with open(os.path.join(config.save_logs, "config.json"), 'w') as gcfg:
            gcfg.write(json.dumps(generator_config)+"\n")



    D_iters = config.D_iters
    for epoch in range(epoch_s,max_epoch):
        train_sampler.set_epoch(epoch) # make shuffling work properly across multiple epochs.
        data_iter = iter(train_dataloader)
        i=0
        if epoch<2:
            image_snapshot_step=500
        else:
            image_snapshot_step=image_snapshot_ticks

        while i<len(train_dataloader):
            j = 0
            while j < D_iters and i < len(train_dataloader):
                j += 1
                i += 1
                data = data_iter.next()
                x_s = data['x_s']
                x_t = data['x_t']
                # print(x_s.shape)
                x_s = x_s.float().cuda(non_blocking=True)
                x_t = x_t.float().cuda(non_blocking=True)

                ############################
                # (1) Update D,E' network
                ############################
                w_s = E(x_s)
                w_t = E(x_t)

                xrec_s = G.net.synthesis(w_s)
                xrec_t = G.net.synthesis(w_t)

                z_s=E_hat(x_s)
                z_t=E_hat(x_t)

                xrec_s_hat = G.net.synthesis(z_s)
                xrec_t_hat = G.net.synthesis(z_t)

                l_s = l_func(xrec_s, xrec_s_hat)
                l_t = l_func(xrec_t, xrec_t_hat)
                dst = torch.mean(l_s) - torch.mean(phistar_gf(l_t))

                x_real=D(x_s)
                x_fake=D(xrec_s.detach())
                loss_real = GAN_loss(x_real, real=True)
                loss_fake = GAN_loss(x_fake, real=False)
                loss_gp = div_loss_(D, x_s, xrec_s.detach())

                optimizer_Ehat.zero_grad()
                optimizer_D.zero_grad()
                adv_loss = loss_real +  loss_fake + 5 * loss_gp
                loss_Ehat = -dst+adv_loss
                loss_Ehat.backward()
                nn.utils.clip_grad_norm_(E_hat.parameters(), 10)
                optimizer_Ehat.step()
                optimizer_D.step()

                Ehat_iterations += 1
                if (Ehat_iterations ) % Ehat_lr_args.decay_step == 0:
                    lr_scheduler_Ehat.step()
                    lr_scheduler_D.step()
                if writer and config.local_rank==0:
                    writer.add_scalar('max/dst', dst.item(), global_step=Ehat_iterations)
                    writer.add_scalar('max/src', l_s.mean().item(), global_step=Ehat_iterations)
                    writer.add_scalar('max/trg', l_t.mean().item(), global_step=Ehat_iterations)
                    writer.add_scalar('max/d_real', loss_real.item(), global_step=Ehat_iterations)
                    writer.add_scalar('max/d_fake', loss_fake.item(), global_step=Ehat_iterations)
                    writer.add_scalar('max/d_gp', loss_gp.item(), global_step=Ehat_iterations)
            ############################
            # (2) Update E,D network
            ############################
            x_s = data['x_s'].float().cuda(non_blocking=True)
            x_t = data['x_t'].float().cuda(non_blocking=True)

            batch_size = x_t.shape[0]
            w_s = E(x_s)
            w_t = E(x_t)

            xrec_s = G.net.synthesis(w_s)
            xrec_t = G.net.synthesis(w_t)

            z_s = E_hat(x_s)
            z_t = E_hat(x_t)

            xrec_s_hat = G.net.synthesis(z_s)
            xrec_t_hat = G.net.synthesis(z_t)

            l_s = l_func(xrec_s, xrec_s_hat)
            l_t = l_func(xrec_t, xrec_t_hat)
            dst = torch.mean(l_s) - torch.mean(phistar_gf(l_t))

            loss_feat = 0
            if loss_feat_weight>0:
                x_feat = F.net(x_s)
                x_rec_feat = F.net(xrec_s)
                loss_feat = torch.mean((x_feat - x_rec_feat) ** 2)
            loss_lpips=0
            if loss_lpi_weight>0:
                loss_lpips=Lpips_loss(x_s,xrec_s)
            loss_adv = 0.
            if loss_adv_weight>0:
                x_adv = D(xrec_s)
                loss_adv = GAN_loss(x_adv, real=True)

            task_loss_pix=mytask_loss_(x_s,xrec_s)

            optimizer_E.zero_grad()
            loss_E=loss_pix_weight*task_loss_pix+loss_dst_weight*dst+loss_feat_weight*loss_feat+loss_lpi_weight*loss_lpips+loss_adv_weight*loss_adv
            loss_E.backward()
            nn.utils.clip_grad_norm_(E.parameters(), 10)
            optimizer_E.step()

            if writer and config.local_rank==0:
                writer.add_scalar('min/pixel', task_loss_pix.item(), global_step=E_iterations)
                writer.add_scalar('min/dst', dst.item(), global_step=E_iterations)
                writer.add_scalar('min/src', l_s.mean().item(), global_step=E_iterations)
                writer.add_scalar('min/trg', l_t.mean().item(), global_step=E_iterations)
                writer.add_scalar('min/vgg',loss_feat.item(),global_step=E_iterations)
                writer.add_scalar('min/adv',loss_adv.item(),global_step=E_iterations)
                # writer.add_scalar('min/lpips',loss_lpips.item(),global_step=E_iterations)
            if config.local_rank==0:
                log_message= f"[Task Loss:(pixel){task_loss_pix.cpu().detach().numpy():.5f}, D {loss_adv.cpu().detach().numpy():.5f},vgg {loss_feat.cpu().detach().numpy()},lpips{loss_lpips}" \
                         f", Fdal Loss:{dst.cpu().detach().numpy():.5f},src:{l_s.mean().cpu().detach().numpy():.5f},trg:{l_t.mean().cpu().detach().numpy():.5f}] "
            if logger and config.local_rank==0 :
                logger.debug(f'Epoch:{epoch:03d}, '
                             f'E_Step:{i:04d}, '
                             f'Dlr:{optimizer_D.state_dict()["param_groups"][0]["lr"]:.2e}, '
                             f'Elr:{optimizer_E.state_dict()["param_groups"][0]["lr"]:.2e}, '
                             f'Ehatlr:{optimizer_Ehat.state_dict()["param_groups"][0]["lr"]:.2e}, '
                             f'{log_message}')
            if (E_iterations % image_snapshot_step == 0) and config.local_rank==0:
                with torch.no_grad():
                    x_train = torch.cat([x_s, xrec_s, x_t, xrec_t], dim=0)
                save_filename = f'train_E_iterations_{E_iterations:05d}.png'
                save_filepath = os.path.join(config.save_images, save_filename)
                tvutils.save_image(tensor=x_train, fp=save_filepath, nrow=batch_size, normalize=True,scale_each=True)
                E.eval()
                for val_step, val_items in enumerate(val_dataloader):
                    if val_step > config.test_save_step:
                        E.train()
                        break
                    with torch.no_grad():
                        x_s = val_items['x_s'].float().cuda(non_blocking=True)
                        x_t = val_items['x_t'].float().cuda(non_blocking=True)

                        batch_size = x_t.shape[0]

                        w_s = E(x_s)
                        w_t = E(x_t)

                        xrec_s = G.net.synthesis(w_s)
                        xrec_t = G.net.synthesis(w_t)
                        loss_pix = torch.mean((x_s - xrec_s) ** 2)
                        x_all = torch.cat([x_s, xrec_s, x_t, xrec_t], dim=0)
                        save_filename = f'epoch_{epoch:03d}_step_{i:04d}_test_{val_step:04d}.png'
                        save_filepath = os.path.join(config.save_images, save_filename)
                        tvutils.save_image(tensor=x_all, fp=save_filepath, nrow=batch_size, normalize=True,scale_each=True)
                    if writer:
                        writer.add_scalar('test/pixel', loss_pix.item(), global_step=E_iterations)

            E_iterations+=1
            if E_iterations % E_lr_args.decay_step == 0 :
                lr_scheduler_E.step()

                    # lr_scheduler_Dhat.step()


        if (epoch+1) %5 == 0 and config.local_rank==0:
            save_filename = f'styleganinv_encoder_epoch_{epoch:03d}.pth'
            save_filepath = os.path.join(config.save_models, save_filename)
            torch.save(E.state_dict(), save_filepath)
            checkpoint = {"E_hat": E_hat.state_dict(),
                          "D":D.state_dict(),
                          "optE_hat": optimizer_Ehat.state_dict(),
                          "optD":optimizer_D.state_dict(),
                          "optE": optimizer_E.state_dict(),
                          "schedulerE": lr_scheduler_E.state_dict(),
                          "schedulerD": lr_scheduler_D.state_dict(),
                          "schedulerDhat": lr_scheduler_Ehat.state_dict(),
                          "epoch": epoch + 1}
            path_checkpoint = "{0}/checkpoint_{1}_epoch.pkl".format(config.save_models, epoch + 1)
            torch.save(obj=checkpoint, f=path_checkpoint)
