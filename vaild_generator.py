#  whether Generator is well pretrained

import os
os.chdir('/home/cy/xing/IDinvert_code/idinvert_pytorch-mycode/') # convenient for debug
import numpy as np
import torch
import torchvision.utils as tvutils
import torchvision
from matplotlib import pyplot
import matplotlib
matplotlib.use('TKAgg')
from utils.logger import setup_logger

from datetime import datetime
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from training.misc import EasyDict
from training.training_loop_encoder import training_loop
from utils.logger import setup_logger
from tensorboardX import SummaryWriter
from models.stylegan_generator import StyleGANGenerator
from models.stylegan_generator2 import G_mapping,G_synthesis


# model_name='stylegan-ffhq-1024'
# model_name='styleganinv_tower256'
model_name='styleganinv_ffhq256'
current_time = datetime.now().strftime('%b%d_%H-%M')
save_root='/home/cy/xing/IDinvert_code/idinvert_pytorch-mycode/output'
save_logs = os.path.join(save_root,  current_time, 'save_logs')
logger = setup_logger(save_logs, 'inversion.log', 'inversion_logger')
logger.info(f'Loading model.')

gpu_ids=None
G = StyleGANGenerator(model_name, logger, gpu_ids=gpu_ids)
G.net.synthesis.eval()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
G.net.to(device)
sample_num=14
z_space_dim=512
# z = np.random.randn(sample_num, z_space_dim)
# z=z.astype(np.float32) #[bn,z,1,1]
bn=4
np.random.seed(2021)
for cnt in range(200):
    latent_code=[]
    for i in range(bn):
        z = np.random.randn(sample_num, z_space_dim).astype(np.float32)
        # z=z[np.newaxis,:]
        latent_code.append(z)
    latent_code=np.array(latent_code) # np array [bn,14,512]
    z_tensor=torch.from_numpy(latent_code)
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    z_tensor=z_tensor.float().cuda()
    with torch.no_grad():
        x_rec = G.net.synthesis(z_tensor)
    #x_rec = (x_rec.clamp(-1, 1) + 1) / 2.0
    # imgs = torchvision.utils.make_grid(x_rec, nrow=4)
    # pyplot.figure(figsize=(15, 6))
    # pyplot.imshow(imgs.permute(1, 2, 0).cpu().detach().numpy())
    # pyplot.show()
    save_filename = f'./temp_data/stylegan_{model_name}__{current_time}_{cnt}.png'
    #save_filepath = os.path.join(config.save_images, save_filename)
    tvutils.save_image(tensor=x_rec, fp=save_filename, nrow=4,normalize=True, scale_each=True)
