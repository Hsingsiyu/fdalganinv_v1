# python 3.6
"""Inverts given images to latent codes with In-Domain GAN Inversion.

Basically, for a particular image (real or synthesized), this script first
employs the domain-guided encoder to produce a initial point in the latent
space and then performs domain-regularized optimization to refine the latent
code.
"""

import os
os.chdir('/home/xsy/idinvert_pytorch-mycode/') # convenient for debug
import argparse
# from tqdm import tqdm
import numpy as np

# from utils.inverter import StyleGANInverter
from utils.logger import setup_logger
from utils.visualizer import HtmlPageVisualizer
from utils.visualizer import save_image, load_image, resize_image

import datasets.celebahq
from models.stylegan_generator import StyleGANGenerator
from models.stylegan_encoder import StyleGANEncoder
from torch.utils.data import DataLoader
import torchvision.utils as tvutils
from tensorboardX import SummaryWriter

import torch
def parse_args():
  """Parses arguments."""
  parser = argparse.ArgumentParser()
  parser.add_argument('--model_name', type=str,default='styleganinv_ffhq256', help='Name of the GAN model.') #todo
  parser.add_argument('--data_root', type=str,default='/home/xsy/FFHQ',
                      help='List of images to invert.')
  parser.add_argument('--image_list', type=str,default='/home/xsy/FFHQ',
                      help='List of images to invert.')
  parser.add_argument('-o', '--output_dir', type=str, default='./val_encoder/real',
                      help='Directory to save the results. If not specified, '
                           '`./results/inversion/${IMAGE_LIST}` '
                           'will be used by default.')
  parser.add_argument('--learning_rate', type=float, default=0.01,
                      help='Learning rate for optimization. (default: 0.01)')
  parser.add_argument('--num_iterations', type=int, default=100,
                      help='Number of optimization iterations. (default: 100)')
  parser.add_argument('--num_results', type=int, default=5,
                      help='Number of intermediate optimization results to '
                           'save for each sample. (default: 5)')
  parser.add_argument('--loss_weight_feat', type=float, default=5e-5,
                      help='The perceptual loss scale for optimization. '
                           '(default: 5e-5)')
  parser.add_argument('--loss_weight_enc', type=float, default=2.0,
                      help='The encoder loss scale for optimization.'
                           '(default: 2.0)')
  parser.add_argument('--viz_size', type=int, default=256,
                      help='Image size for visualization. (default: 256)')
  parser.add_argument('--gpu_id', type=str, default='0',
                      help='Which GPU(s) to use. (default: `0`)')
  parser.add_argument('--image_size', type=int, default=256,
                      help='the image size in training dataset (defaults; 256)')
  parser.add_argument('--batch_size', type=int, default=1,
                      help='the batch size in one picture ')
  return parser.parse_args()


def main():
  """Main function."""
  args = parse_args()
  os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
  # assert os.path.exists(args.data_root)
  # img_list=os.listdir(args.image_list)
  image_list_name = os.path.splitext(os.path.basename(args.data_root))[0]
  output_dir = args.output_dir or f'results/inversion/{image_list_name}'
  logger = setup_logger(output_dir, 'inversion.log', 'inversion_logger')

  logger.info(f'Loading model.')
  # writer = SummaryWriter(os.path.join(output_dir, 'pix_loss_inFFHQ'))
  class Config:
      data_root = args.data_root
      size = args.image_size
      min_val = -1.0
      max_val = 1.0
      split = 3500
  dataset_args = Config()

  # Load image list.
  val_dataset = datasets.celebahq.FFHQDataset(dataset_args, train=True)#todo
  # train_dataset = datasets.celebahq.FFHQDataset(dataset_args, train=True)

  # train_dataloader = DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True)
  val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)


  #
  G = StyleGANGenerator(args.model_name, logger, gpu_ids=None)
  E = StyleGANEncoder(args.model_name, logger, gpu_ids=None)
  # E1=StyleGANEncoder(args.model_name,logger,gpu_ids=None)

  #load
  weight_path='/home/xsy/idinvert_pytorch-mycode/idinvert_output/ffhq-InitEncoder_StyleDJan28_20-58_retrain_bs_48_epoch2000_D_fmap_max128_datasplit3500/save_models/epoch_1990_step_0071_test_0003.pth'
  checkpoint=torch.load(weight_path)
  E.net.load_state_dict(checkpoint["E"])
  G.net.synthesis.eval()
  E.net.eval()
  encode_dim = [G.num_layers, G.w_space_dim]
  global_step=0
  x_real=[]
  x_fake=[]
  for  step, items in enumerate(val_dataloader):
      global_step=global_step+1
      x = items
      x = x.float().cuda()
      batch_size = x.shape[0]
      with torch.no_grad():
          z = E.net(x).view(batch_size, *encode_dim)
          x_rec = G.net.synthesis(z)
          # pix_loss=((x-x_rec)**2).mean()
      x_real.append(x)
      x_fake.append(x_rec)

      # writer.add_scalar('pixel_loss', pix_loss.item(), global_step=global_step)
      # x_all = torch.cat([x,x_rec], dim=0)
      # save_filename = f'step_{step:05d}.png'
      # save_filepath = os.path.join(output_dir, save_filename)
      # tvutils.save_image(tensor=x, fp=save_filepath,  normalize=True, scale_each=True)
      # print(f'{step}/{len(val_dataloader)}:')
      # print(pix_loss)

  x1=torch.cat(x_real,dim=0)
  x2=tor.cat(x_fake,dim=0)
  out = swd(x1, x2, device="cuda")
  print(out)
if __name__ == '__main__':
  main()
