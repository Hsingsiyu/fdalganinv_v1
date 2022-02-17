#只训练D
import argparse
import os
import numpy as np
import torchvision.transforms as transforms
from torchvision.utils import save_image
os.chdir('/home/xsy/idinvert_pytorch-mycode')
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

from torch.utils.data import DataLoader
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch
import datasets.celebahq
from models.stylegan_generator_network import StyleGANGeneratorNet
from models.stylegan_discriminator_network2 import StyleGANDiscriminator
from tensorboardX import SummaryWriter
from datetime import datetime
import random

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
def GAN_loss(scores_out, real=True):
    if real:
        return torch.mean(F.softplus(-scores_out))
    else:
        return torch.mean(F.softplus(scores_out))
def div_loss_(D, real_x,  p=2, cuda=False):
    x_ = real_x.requires_grad_(True)
    y_ = D(x_)
    # cal f'(x)
    grad = torch.autograd.grad(
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
def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
       # nn.init.constant_(m.bias, 0)
    # 也可以判断是否为conv2d，使用相应的初始化方式
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
     # 是否为批归一化层
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
# os.makedirs("images", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=32, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.001, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.0, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.99, help="adam: decay of first order momentum of gradient")
parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")
parser.add_argument("--image_size", type=int, default=256, help="interval betwen image samples")
parser.add_argument("--data_root", type=str, default='/home/xsy/FFHQ', help="path")
parser.add_argument("--save_step", type=int, default=50)
parser.add_argument('--save_root', type=str, default='/home/xsy/idinvert_pytorch-mycode/trainStyleD_output/')

opt = parser.parse_args()
print(opt)
prefix = 'train_StyleD'
current_time = datetime.now().strftime('%b%d_%H-%M')
opt.save_images = os.path.join(opt.save_root,  prefix+current_time, 'save_images')
opt.save_models = os.path.join(opt.save_root,  prefix+current_time, 'save_models')
writer = SummaryWriter(os.path.join(opt.save_root,prefix+current_time))
if not os.path.exists(opt.save_images):
    os.makedirs(opt.save_images)
if not os.path.exists(opt.save_models):
    os.makedirs(opt.save_models)

cuda = True if torch.cuda.is_available() else False
setup_seed(42)
##data loader
class Config:
    data_root = opt.data_root
    size = opt.image_size
    min_val = -1.0
    max_val = 1.0
    split = 20000  # 65000
dataset_args = Config()
train_dataset = datasets.celebahq.FFHQDataset(dataset_args, train=True)
dataloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True,drop_last=True)


# Initialize generator and discriminator
generator = StyleGANGeneratorNet(opt.image_size,repeat_w=False,final_tanh=True)
discriminator = StyleGANDiscriminator(opt.image_size,fmaps_max=128) #todo
state_dict=torch.load('./models/pretrain/styleganinv_ffhq256_generator.pth')
state_dict['truncation.truncation']=generator.state_dict()['truncation.truncation']
generator.load_state_dict(state_dict)
discriminator.apply(weight_init)

generator.eval()
discriminator.train()
# print(generator)
if cuda:
    generator.cuda()
    discriminator.cuda()
    # adversarial_loss.cuda()

# Optimizers
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2), eps=1e-8)
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# ----------
#  Training
# ----------
global_step=0
for epoch in range(opt.n_epochs):
    for i, (imgs) in enumerate(dataloader):

        # Adversarial ground truths
        bs=imgs.size(0)
        encode_dim = [bs, generator.num_layers, generator.w_space_dim]

        valid = Variable(Tensor(imgs.size(0), 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(imgs.size(0), 1).fill_(0.0), requires_grad=False)

        # Configure input
        real_imgs = Variable(imgs.type(Tensor))

        # Sample noise as generator input
        # z = Variable(Tensor(np.random.normal(0, 1, encode_dim)))
        z = Variable(Tensor(bs,512).normal_(0,1))
        w=generator.mapping(z).view(encode_dim)
        gen_imgs = generator.synthesis(w)
        #  Generate a batch of images

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()
        # Measure discriminator's ability to classify real from generated samples
        real_loss = GAN_loss(discriminator(real_imgs), True)
        fake_loss = GAN_loss(discriminator(gen_imgs.detach()), False)
        d_gp=div_loss_(discriminator,real_imgs,cuda=cuda)
        d_loss = (real_loss + fake_loss) / 2+2.5*d_gp #与IDinvert的超参数一致
        d_loss.backward()
        optimizer_D.step()
        global_step += 1

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss real:%f fake:%f gp:%f] "
            % (epoch, opt.n_epochs, i, len(dataloader), real_loss.item(), fake_loss.item(),d_gp.item()) #todo
        )

        batches_done = epoch * len(dataloader) + i
        if batches_done % opt.sample_interval == 0:
            imgfilename = f'step_{batches_done:04d}.png'
            save_image(gen_imgs.data[:25, :, :, :], os.path.join(opt.save_images, imgfilename), nrow=5, normalize=True)
        #writer
        if writer:
            # writer.add_scalar('D/grad',grad_mean/l, global_step=global_step)
            writer.add_scalar('D/loss_real', real_loss.item(), global_step=global_step)
            writer.add_scalar('D/loss_fake', fake_loss.item(), global_step=global_step)
            writer.add_scalar('D/loss_gp', d_gp.item(), global_step=global_step)
            writer.add_scalar('D/loss', d_loss.item(), global_step=global_step)
        # todo optimizer shchedule
    if (epoch+1)%opt.save_step==0:
        save_filename = f'styleganffhq256_discriminator_epoch_{epoch:03d}'
        save_filepath = os.path.join(opt.save_root, save_filename)
        torch.save(discriminator.state_dict(), save_filepath)