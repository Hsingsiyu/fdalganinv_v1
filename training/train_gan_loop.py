import datasets.celebahq
from models.stylegan_generator import StyleGANGenerator
from models.perceptual_model import PerceptualModel
from models.stylegan_encoder import StyleGANEncoder
from models.stylegan_discriminator_network2 import h_layers,StyleGANDiscriminator
from training.misc import EasyDict

import os
import torch
import torch.nn as nn
import torchvision.utils as tvutils
from torch.utils.data import DataLoader
import  copy
import json
from fDAL.utils import ConjugateDualFunction
from torch.nn.parallel import DistributedDataParallel as DDP
import itertools
import random
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
def div_loss_(D, real_x):
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
    div = (grad * grad).sum(dim=1, keepdim=True).sum(dim=2, keepdim=True).sum(dim=3, keepdim=True)
    div = torch.mean(div)
    return div

def training_loop(
        config,
        dataset_args={},
        E_lr_args=EasyDict(),
        D_lr_args=EasyDict(),
        Dhat_lr_args=EasyDict(),
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
    Dhat_iterations=0

    loss_pix_weight=loss_args.loss_pix_weight
    loss_w_weight=loss_args.loss_w_weight
    loss_dst_weight=loss_args.loss_dst_weight
    loss_feat_weight=loss_args.loss_feat_weight
    loss_adv_weight=0.1

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

    # construct model
    G = StyleGANGenerator(config.model_name, logger, gpu_ids=config.gpu_ids,local_rank=config.local_rank)
    E = StyleGANEncoder(config.model_name, logger, gpu_ids=config.gpu_ids,local_rank=config.local_rank)
    F = PerceptualModel(min_val=-1, max_val=1, gpu_ids=config.gpu_ids,local_rank=config.local_rank)
    Discri = StyleGANDiscriminator(config.image_size, fmaps_max=128).cuda()
    F.net.eval()
    D=h_layers(config.image_size,fmaps_max=128)
    D_hat = copy.deepcopy(D)
    # [bn,128,h/16.w/16]

    D.cuda()
    D_hat.cuda()
    #DDP前先送入GPU
    if config.gpu_ids is not None:
        assert len(config.gpu_ids) > 1
        D = DDP(D, device_ids=[config.local_rank],broadcast_buffers=False, find_unused_parameters=True)
        D_hat =DDP(D_hat, device_ids=[config.local_rank],broadcast_buffers=False, find_unused_parameters=True)
        Discri=DDP(Discri, device_ids=[config.local_rank],broadcast_buffers=False, find_unused_parameters=True)

    # load parameter
    E.net.apply(weight_init)
    Discri.apply(weight_init)
    D.apply(weight_init)
    D_hat.apply(lambda self_: self_.reset_parameters() if hasattr(self_, 'reset_parameters') else None)

    for p in E.net.parameters():
        p.requires_grad = True

    Discri.train()
    E.net.train()
    G.net.synthesis.eval()
    encode_dim = [G.num_layers, G.w_space_dim]

    # setup optimizer
    optimizer_Discri=torch.optim.Adam(Discri.parameters(), lr=D_lr_args.learning_rate, **opt_args)
    optimizer_E = torch.optim.Adam(E.net.parameters(), lr=E_lr_args.learning_rate, **opt_args)
    optimizer_D = torch.optim.Adam(D.parameters(), lr=D_lr_args.learning_rate, **opt_args)

    if config.adam:
        optimizer_Dhat =torch.optim.Adam(D_hat.parameters(), lr=Dhat_lr_args.learning_rate, **opt_args)
    else:
        optimizer_Dhat = torch.optim.SGD(D_hat.parameters(), lr=Dhat_lr_args.learning_rate, momentum=0.9,
                                         nesterov=True, weight_decay=0.02)

    if config.netE!='':
        with torch.no_grad():
            E.net.load_state_dict(torch.load(config.netE,map_location=torch.device("cpu")))

    if config.netD_hat!='':
        with torch.no_grad():
            checkpoint=torch.load(config.netD_hat,map_location=torch.device('cpu'))
        D_hat.load_state_dict(checkpoint["h_hat"])
        D.load_state_dict(checkpoint["h"])
        Discri.load_state_dict(checkpoint["D"])
        optimizer_Discri.load_state_dict(checkpoint["opt_Discri"])
        optimizer_Dhat.load_state_dict(checkpoint["optD_hat"])
        optimizer_E.load_state_dict((checkpoint["optE"]))
        optimizer_D.load_state_dict(checkpoint["optD"])
        epoch_s=checkpoint["epoch"]

    #特别详细的参数选择的记录
    lr_scheduler_Discri=torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer_Discri, gamma=Dhat_lr_args.decay_rate)
    lr_scheduler_E = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer_E, gamma=E_lr_args.decay_rate)
    lr_scheduler_Dhat=torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer_Dhat, gamma=Dhat_lr_args.decay_rate)
    lr_scheduler_D = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer_D, gamma=D_lr_args.decay_rate)
    if config.netD_hat!='':
        lr_scheduler_E.load_state_dict(checkpoint['schedulerE'])
        lr_scheduler_D.load_state_dict(checkpoint['schedulerD'])
        lr_scheduler_D.load_state_dict(checkpoint['schedulerD'])
        lr_scheduler_Discri.load_state_dict(checkpoint['schedulerDiscri'])
    if config.local_rank==0:
        generator_config = {"imageSize": config.image_size, "dataset": config.dataset_name, "Dhat_iters":config.D_iters,"trainingset":dataset_args.split,
                            "train_bs": config.train_batch_size,"val_bs": config.test_batch_size,"div":config.divergence,"nepoch":config.nepoch,
                            "adam":config.adam,"lr_E":optimizer_E.state_dict()["param_groups"][0]["lr"],
                            "lr_D": optimizer_D.state_dict()["param_groups"][0]["lr"],"lr_Dhat":optimizer_Dhat.state_dict()["param_groups"][0]["lr"],
                            "pix_weight":loss_args.loss_pix_weight,"w_weight":loss_args.loss_w_weight,"dst_weight":loss_args.loss_dst_weight,"loss_adv_weight":loss_adv_weight,
                            "Edecay_rate":E_lr_args.decay_step,"Ddecay_rate":D_lr_args.decay_step,  "Dhat_decay_rate":Dhat_lr_args.decay_step,
                            }
        with open(os.path.join(config.save_logs, "config.json"), 'w') as gcfg:
            gcfg.write(json.dumps(generator_config)+"\n")

    image_snapshot_step = image_snapshot_ticks

    l_func = nn.L1Loss(reduction='none')
    phistar_gf = lambda t: ConjugateDualFunction(config.divergence).fstarT(t)

    D_iters = config.D_iters
    for epoch in range(epoch_s,max_epoch):
        train_sampler.set_epoch(epoch)
        data_iter = iter(train_dataloader)
        i=0
        if epoch<1:
            image_snapshot_step=50
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
                x_s = x_s.float().cuda(non_blocking=True)
                x_t = x_t.float().cuda(non_blocking=True)
                batch_size = x_t.shape[0]
                ############################
                # (1) Update D' network
                ############################
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

                Discri_loss=0
                if loss_adv_weight>0:
                    x_real=Discri(x_s)
                    x_fake=Discri(xrec_s.detach())
                    loss_real = GAN_loss(x_real, real=True)
                    loss_fake = GAN_loss(x_fake, real=False)
                    loss_gp = div_loss_(Discri, x_s)
                    Discri_loss = 1 * loss_real + 1 * loss_fake + 5 * loss_gp
                loss_all = 0.0
                grad_D=torch.autograd.grad(outputs=Discri_loss,inputs=Discri.parameters(),create_graph=False,retain_graph=True,allow_unused=True)[0]
                gradnorm_D= torch.norm(grad_D, dim=None)

                grad_hhat=torch.autograd.grad(outputs=dst,inputs=D_hat.parameters(),create_graph=False,retain_graph=True,allow_unused=True)[0]
                gradnorm_hhat= torch.norm(grad_hhat, dim=None)

                if gradnorm_D>0:
                    loss_all+=torch.div(input=Discri_loss,other=gradnorm_D.detach())
                if gradnorm_hhat>0:
                    loss_all+=torch.div(input=dst,other=gradnorm_hhat.detach())
                optimizer_Dhat.zero_grad()
                optimizer_Discri.zero_grad()
                loss_all.backward()
                # loss_Dhat.backward()
                # nn.utils.clip_grad_norm_(D_hat.parameters(), 10)
                optimizer_Dhat.step()
                optimizer_Discri.step()
                Dhat_iterations += 1
                if (Dhat_iterations ) % Dhat_lr_args.decay_step == 0:
                    lr_scheduler_Dhat.step()
                    lr_scheduler_Discri.step()
                if writer and config.local_rank==0:
                    writer.add_scalar('max/normD', gradnorm_D.item(), global_step=Dhat_iterations)
                    writer.add_scalar('max/normh', gradnorm_hhat.item(), global_step=Dhat_iterations)
                    writer.add_scalar('max/loss_real', loss_real.item(), global_step=Dhat_iterations)
                    writer.add_scalar('max/loss_fake', loss_fake.item(), global_step=Dhat_iterations)
                    writer.add_scalar('max/loss_gp', loss_gp.item(), global_step=Dhat_iterations)
                    writer.add_scalar('max/loss', Discri_loss.item(), global_step=Dhat_iterations)
                    writer.add_scalar('max/dst', dst.item(), global_step=Dhat_iterations)
                    writer.add_scalar('max/src', l_s.mean().item(), global_step=Dhat_iterations)
                    writer.add_scalar('max/trg', l_t.mean().item(), global_step=Dhat_iterations)
            ############################
            # (2) Update E,D network
            ############################
            x_s = data['x_s']
            x_t = data['x_t']
            x_s = x_s.float().cuda(non_blocking=True)
            x_t = x_t.float().cuda(non_blocking=True)
            batch_size = x_t.shape[0]

            w_s=E.net(x_s).view(batch_size, *encode_dim)
            w_t=E.net(x_t).view(batch_size, *encode_dim)

            xrec_s = G.net.synthesis(w_s)
            xrec_t = G.net.synthesis(w_t)

            source_label = D(x_s)  # h(x_s)
            features_s = D(xrec_s)
            features_t = D(xrec_t)

            loss_all = 0.
            loss_feat = 0.
            if loss_feat_weight>0:
                x_feat = F.net(x_s)
                x_rec_feat = F.net(xrec_s)
                loss_feat = torch.mean((x_feat - x_rec_feat) ** 2)
                # grad_feat=torch.autograd.grad(outputs=loss_feat,inputs=E.net.parameters(),retain_graph=True,only_inputs=True,allow_unused=True,create_graph=True,)[0]
            # grad_vgg=torch.autograd.grad(outputs=loss_feat, inputs= E.net.parameters(), create_graph=False, retain_graph=True,allow_unused=True)[0]
            # gradnorm_vgg=torch.norm(grad_vgg)
            if loss_feat_weight>0:
                loss_all+=0.00005*loss_feat #torch.div(loss_feat,gradnorm_vgg.detach())

            # task loss in pixel and code
            task_loss_w = mytask_loss_(features_s, source_label)  # L(h(x),hGE(x))
            grad_w=torch.autograd.grad(outputs=task_loss_w, inputs=itertools.chain(E.net.parameters(),D.parameters()), create_graph=False, retain_graph=True,allow_unused=True)[0]
            gradnorm_w=torch.norm(grad_w)
            if gradnorm_w>0:
                loss_all+=torch.div(task_loss_w,gradnorm_w.detach())
            print(gradnorm_w)
            task_loss_pix = mytask_loss_(x_s.detach(), xrec_s)  # L(x_s,G(E(x_s)))
            grad_pix=torch.autograd.grad(outputs=task_loss_pix, inputs=E.net.parameters(), create_graph=False, retain_graph=True,allow_unused=True)[0]
            gradnorm_pix=torch.norm(grad_pix)
            if gradnorm_pix>0:
                loss_all+=torch.div(task_loss_pix,gradnorm_pix.detach())
            print(gradnorm_pix)


            features_s_adv = D_hat(xrec_s)  # h'(GE(x_s))
            features_t_adv = D_hat(xrec_t)  # h'(GE(x_t))

            l_s = l_func(features_s_adv, features_s)
            l_t = l_func(features_t_adv, features_t)
            dst =  torch.mean(l_s) - torch.mean(phistar_gf(l_t))
            grad_dst =torch.autograd.grad(outputs=dst, inputs=itertools.chain(E.net.parameters(),D.parameters()), create_graph=False, retain_graph=True,
                                allow_unused=True)[0]
            gradnorm_dst = torch.norm(grad_dst, dim=None)
            if loss_dst_weight>0:
                loss_all+=torch.div(dst,gradnorm_dst.detach())
            print(gradnorm_dst)

            loss_adv = 0.
            if loss_adv_weight>0:
                x_adv = Discri(xrec_s)
                loss_adv = GAN_loss(x_adv, real=True)
            grad_adv=torch.autograd.grad(outputs=loss_adv, inputs=E.net.parameters(), create_graph=False, retain_graph=True,allow_unused=True)[0]
            gradnorm_adv=torch.norm(grad_adv)
            if gradnorm_adv>0:
                loss_all+=torch.div(loss_adv,gradnorm_adv.detach())
            print(gradnorm_adv)

            optimizer_E.zero_grad()
            optimizer_D.zero_grad()
            # loss_E=loss_pix_weight*task_loss_pix+loss_w_weight*task_loss_w+loss_dst_weight*dst+loss_feat_weight*loss_feat+loss_adv_weight*loss_adv
            # loss_E.backward()
            loss_all.backward()
            # nn.utils.clip_grad_norm_(D.parameters(), 10)
            optimizer_E.step()
            optimizer_D.step()

            if writer and config.local_rank==0:
                writer.add_scalar('min/adv', loss_adv.item(), global_step=E_iterations)
                writer.add_scalar('min/pixel', task_loss_pix.item(), global_step=E_iterations)
                writer.add_scalar('min/h',task_loss_w.item(), global_step=E_iterations)
                writer.add_scalar('min/dst', dst.item(), global_step=E_iterations)
                writer.add_scalar('min/src', l_s.mean().item(), global_step=E_iterations)
                writer.add_scalar('min/trg', l_t.mean().item(), global_step=E_iterations)
                writer.add_scalar('min/vgg',loss_feat.item(),global_step=E_iterations)
                writer.add_scalar('min/gradpix',gradnorm_pix.item(),global_step=E_iterations)
                writer.add_scalar('min/gradw',gradnorm_w,global_step=E_iterations)
                writer.add_scalar('min/grad_adv',gradnorm_adv,global_step=E_iterations)
                writer.add_scalar('min/grad_dst',gradnorm_dst,global_step=E_iterations)
                # writer.add_scalar('min/gradvgg',gradnorm_vgg,global_step=E_iterations)

            if  config.local_rank==0:
                log_message= f"[Task Loss:(pixel){task_loss_pix.cpu().detach().numpy():.5f}, h {task_loss_w.cpu().detach().numpy():.5f},vgg {loss_feat.cpu().detach().numpy()}" \
                         f", Fdal Loss:{dst.cpu().detach().numpy():.5f},src:{l_s.mean().cpu().detach().numpy():.5f},trg:{l_t.mean().cpu().detach().numpy():.5f}] "
            if logger and config.local_rank==0 :
                logger.debug(f'Epoch:{epoch:03d}, '
                             f'E_Step:{i:04d}, '
                             f'Dlr:{optimizer_D.state_dict()["param_groups"][0]["lr"]:.2e}, '
                             f'Elr:{optimizer_E.state_dict()["param_groups"][0]["lr"]:.2e}, '
                             f'Dhatlr:{optimizer_Dhat.state_dict()["param_groups"][0]["lr"]:.2e}, '
                             f'{log_message}')
            if (E_iterations % image_snapshot_step == 0) and config.local_rank==0:
                with torch.no_grad():
                    x_train = torch.cat([x_s, xrec_s, x_t, xrec_t], dim=0)
                save_filename = f'train_E_iterations_{E_iterations:05d}.png'
                save_filepath = os.path.join(config.save_images, save_filename)
                tvutils.save_image(tensor=x_train, fp=save_filepath, nrow=batch_size, normalize=True,
                                   scale_each=True)
                for val_step, val_items in enumerate(val_dataloader):
                    if val_step > config.test_save_step:
                        E.net.train()
                        break
                    with torch.no_grad():
                        E.net.eval()
                        x_s = val_items['x_s']
                        x_t = val_items['x_t']

                        x_s = x_s.float().cuda()
                        x_t = x_t.float().cuda()

                        batch_size = x_t.shape[0]

                        w_s = E.net(x_s).view(batch_size, *encode_dim)
                        w_t = E.net(x_t).view(batch_size, *encode_dim)

                        xrec_s = G.net.synthesis(w_s)
                        xrec_t = G.net.synthesis(w_t)
                        loss_pix = torch.mean((x_s - xrec_s) ** 2)

                        x_all = torch.cat([x_s, xrec_s, x_t, xrec_t], dim=0)
                        save_filename = f'epoch_{epoch:03d}_step_{i:04d}_test_{val_step:04d}.png'
                        save_filepath = os.path.join(config.save_images, save_filename)
                        tvutils.save_image(tensor=x_all, fp=save_filepath, nrow=batch_size, normalize=True,
                                       scale_each=True)
                        if writer:
                            writer.add_scalar('test/L2', loss_pix.item(), global_step=E_iterations)

            if (E_iterations) % E_lr_args.decay_step == 0 :
                lr_scheduler_E.step()
                lr_scheduler_D.step()
            E_iterations += 1
                    # lr_scheduler_Dhat.step()


        if (epoch+1) %5 == 0 and config.local_rank==0:
            save_filename = f'styleganinv_encoder_epoch_{epoch+1:03d}.pth'
            save_filepath = os.path.join(config.save_models, save_filename)
            torch.save(E.net.state_dict(), save_filepath)
            checkpoint = {"h_hat": D_hat.state_dict(),
                          "h":D.state_dict(),
                          "Discri":Discri.state_dict(),
                          "opt_Discri":optimizer_Discri.state_dict(),
                          "optD_hat": optimizer_Dhat.state_dict(),
                          "optD":optimizer_D.state_dict(),
                          "optE": optimizer_E.state_dict(),
                          "schedulerDiscri":lr_scheduler_Discri.state_dict(),
                          "schedulerE": lr_scheduler_E.state_dict(),
                          "schedulerD": lr_scheduler_D.state_dict(),
                          "schedulerDhat": lr_scheduler_Dhat.state_dict(),
                          "epoch": epoch + 1}
            path_checkpoint = "{0}/checkpoint_{1}_epoch.pkl".format(config.save_models, epoch + 1)
            torch.save(obj=checkpoint, f=path_checkpoint)
