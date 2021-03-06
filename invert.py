# python 3.6
"""Inverts given images to latent codes with In-Domain GAN Inversion.

Basically, for a particular image (real or synthesized), this script first
employs the domain-guided encoder to produce a initial point in the latent
space and then performs domain-regularized optimization to refine the latent
code.
"""

import os
import argparse
from tqdm import tqdm
import numpy as np

from utils.inverter import StyleGANInverter
from utils.logger import setup_logger
from utils.visualizer import HtmlPageVisualizer
from utils.visualizer import save_image, load_image, resize_image


def parse_args():
  """Parses arguments."""
  parser = argparse.ArgumentParser()
  parser.add_argument('model_name', default='styleganinv_ffhq256',type=str, help='Name of the GAN model.')
  parser.add_argument('image_list', type=str,default='./examples/ffhq_inv_src.txt',
                      help='List of images to invert.')
  parser.add_argument('-o', '--output_dir', type=str, default='',
                      help='Directory to save the results. If not specified, '
                           '`./results/inversion/${IMAGE_LIST}` '
                           'will be used by default.')
  parser.add_argument('--learning_rate', type=float, default=0.01,
                      help='Learning rate for optimization. (default: 0.01)')
  parser.add_argument('--num_iterations', type=int, default=0,
                      help='Number of optimization iterations. (default: 100)')
  parser.add_argument('--num_results', type=int, default=5,
                      help='Number of intermediate optimization results to '
                           'save for each sample. (default: 5)')
  parser.add_argument('--loss_weight_feat', type=float, default=0.8,
                      help='The perceptual loss scale for optimization. '
                           '(default: 5e-5,0.8 for lpips)')
  parser.add_argument('--loss_weight_enc', type=float, default=2.0,
                      help='The encoder loss scale for optimization.'
                           '(default: 2.0)')
  parser.add_argument('--viz_size', type=int, default=256,
                      help='Image size for visualization. (default: 256)')
  parser.add_argument('--gpu_id', type=str, default='0',
                      help='Which GPU(s) to use. (default: `0`)')
  parser.add_argument('--E_type', type=str, default='baseline',
                      help='baseline or ours')
  parser.add_argument('--netE', type=str, default='/home/xsy/idinvert_pytorch-mycode/fdaloutput/fDAL-FFHQ-InitEncoder_StyleDApr15_14-05_clipgrad_pretrain_bs_16_epoch3000_regcoef5.0_1_1.0_adamTrue_DIV_pearson/save_models/styleganinv_encoder_epoch_899.pth',
                      help='path of encoder')
  return parser.parse_args()


def main():
  """Main function."""
  args = parse_args()
  os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
  assert os.path.exists(args.image_list)
  image_list_name = os.path.splitext(os.path.basename(args.image_list))[0]
  net_name=os.path.splitext(os.path.basename(args.netE))[0]

  output_dir = args.output_dir or f'results/inversion_{args.E_type}/{image_list_name}_{net_name}'
  logger = setup_logger(output_dir, 'inversion.log', 'inversion_logger')

  logger.info(f'Loading model.')
  inverter = StyleGANInverter(
      args.model_name,
      learning_rate=args.learning_rate,
      iteration=args.num_iterations,
      reconstruction_loss_weight=1.0,
      lpips_loss_weight=args.loss_weight_feat,
      regularization_loss_weight=args.loss_weight_enc,
      logger=logger,
      E_type=args.E_type,
      netE=args.netE)
  image_size = inverter.G.resolution

  # Load image list.
  logger.info(f'Loading image list.')
  image_list = []
  with open(args.image_list, 'r') as f:
    for line in f:
      image_list.append(line.strip())

  # Initialize visualizer.
  save_interval = args.num_iterations // args.num_results
  headers = ['Name', 'Original Image', 'Encoder Output']
  for step in range(1, args.num_iterations + 1):
    if step == args.num_iterations or step % save_interval == 0:
      headers.append(f'Step {step:06d}')
  viz_size = None if args.viz_size == 0 else args.viz_size
  visualizer = HtmlPageVisualizer(
      num_rows=len(image_list), num_cols=len(headers), viz_size=viz_size)
  visualizer.set_headers(headers)

  # Invert images.
  logger.info(f'Start inversion.')
  latent_codes = []
  for img_idx in tqdm(range(len(image_list)), leave=False):
    image_path = image_list[img_idx]
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    image = resize_image(load_image(image_path), (image_size, image_size))
    if args.num_iterations>0:
        code, viz_results = inverter.easy_invert(image, num_viz=args.num_results)
    else:
        code, viz_results=inverter.easy_inti_code(image)
    latent_codes.append(code)
    save_image(f'{output_dir}/{image_name}_ori.png', image)
    if args.num_iterations > 0:
        save_image(f'{output_dir}/{image_name}_enc.png', viz_results[1])
    else:
        save_image(f'{output_dir}/{image_name}_enc.png', viz_results[0])

    visualizer.set_cell(img_idx, 0, text=image_name)
    visualizer.set_cell(img_idx, 1, image=image)
    visualizer.set_cell(img_idx, 2, image=viz_results[0])
    if args.num_iterations>0:
        save_image(f'{output_dir}/{image_name}_inv.png', viz_results[-1])
        for viz_idx, viz_img in enumerate(viz_results[1:]):
          visualizer.set_cell(img_idx, viz_idx + 2, image=viz_img)

  # Save results.
  os.system(f'cp {args.image_list} {output_dir}/image_list.txt')
  np.save(f'{output_dir}/inverted_codes.npy',
          np.concatenate(latent_codes, axis=0))
  visualizer.save(f'{output_dir}/inversion.html')


if __name__ == '__main__':
  main()
