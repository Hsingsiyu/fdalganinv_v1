# --coding:utf-8--
import os
os.chdir('/home/xsy/idinvert_pytorch-mycode/') # convenient for debug
import argparse
from datetime import datetime
os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'

from training.misc import EasyDict
from training.trainingloop_e2style import training_loop
from utils.logger import setup_logger
from tensorboardX import SummaryWriter

def main():
    parser = argparse.ArgumentParser(description='Training the in-domain encoder with fDAL')
    parser.add_argument('--data_root', type=str, default='/home/xsy/FFHQ/',
                        help='path to training data (.txt path file)')
    parser.add_argument('--image_size', type=int, default=256,
                        help='the image size in training dataset (defaults; 256)')
    parser.add_argument('--model_name', type=str, default='styleganinv_ffhq256',
                        help='the name of the model')
    parser.add_argument('--dataset_name', type=str, default='ffhq',
                        help='the name of the training dataset (defaults; ffhq)')
    parser.add_argument('--train_batch_size', type=int, default=20,
                        help='training batch size')
    parser.add_argument('--test_batch_size', type=int, default=12,
                        help='training batch size')
    parser.add_argument('--gpu_ids', type=list, default=[0,1],
                        help='list of gpus')
    parser.add_argument('--test_save_step', type=int, default=0,
                        help='how much step to be saved when inference')
    parser.add_argument('--save_root', type=str, default='/home/xsy/idinvert_pytorch-mycode/fdaloutput/')
    parser.add_argument('--divergence', type=str, default='pearson',help='pearson,kl')
    parser.add_argument('--nepoch', type=int, default=3000)
    parser.add_argument('--lrDhat', type=int, default=0.00001)
    parser.add_argument('--lrE', type=int, default=0.00001)
    parser.add_argument('--lrD', type=int, default=0.00001)
    parser.add_argument('--adam', type=bool, default=True)
    parser.add_argument('--D_iters', type=int, default=1)
    parser.add_argument('--netE', type=str, default='/home/xsy/idinvert_pytorch-mycode/fdaloutput/fDAL-FFHQ-e2StyleEncoder_StyleDApr01_09-46_clipgrad_bs_20_epoch3000_regcoef1.0_10_1.0_adamTrue_DIV_pearson/save_models/styleganinv_encoder_epoch_014.pth')
    parser.add_argument('--netD_hat', type=str, default='/home/xsy/idinvert_pytorch-mycode/fdaloutput/fDAL-FFHQ-e2StyleEncoder_StyleDApr01_09-46_clipgrad_bs_20_epoch3000_regcoef1.0_10_1.0_adamTrue_DIV_pearson/save_models/checkpoint_15_epoch.pkl')
    parser.add_argument('--local_rank', type=int, default=0,help='node rank for distributed training')
    args = parser.parse_args()

    class Config:
        data_root = args.data_root
        size = args.image_size
        min_val = -1.0
        max_val = 1.0
        split=50000#65000
    datasets_args = Config()

    loss_args=EasyDict(loss_pix_weight=1.0,loss_w_weight=10,loss_dst_weight=1.0,loss_feat_weight=0,loss_lpi_weight=0.8) #0.5
    opt_args = EasyDict(betas=(0.9, 0.99), eps=1e-8)
    E_lr_args = EasyDict(learning_rate=args.lrE, decay_step=3000, decay_rate=0.8, stair=False)
    D_lr_args = EasyDict(learning_rate=args.lrD, decay_step=3000, decay_rate=0.8, stair=False)
    Dhat_lr_args = EasyDict(learning_rate=args.lrDhat, decay_step=3000, decay_rate=0.8, stair=False)

    current_time = datetime.now().strftime('%b%d_%H-%M')
    prefix = 'fDAL-FFHQ-e2StyleEncoder_StyleD'
    parm='_clipgrad_bs_%s_epoch%s_regcoef%s_%s_%s_adam%s_DIV_%s'%(args.train_batch_size,args.nepoch,loss_args.loss_pix_weight,loss_args.loss_w_weight,loss_args.loss_dst_weight,args.adam,args.divergence)
    args.save_images = os.path.join(args.save_root, prefix  + current_time+parm, 'save_images')
    args.save_models = os.path.join(args.save_root, prefix + current_time+parm, 'save_models')
    args.save_logs = os.path.join(args.save_root, prefix + current_time+parm, 'save_logs')

    if not os.path.exists(args.save_images):
        os.makedirs(args.save_images)
    if not os.path.exists(args.save_models):
        os.makedirs(args.save_models)
    if not os.path.exists(args.save_logs):
        os.makedirs(args.save_logs)
    writer = SummaryWriter(os.path.join(args.save_root, prefix + current_time+parm))

    logger = setup_logger(args.save_logs, 'inversion.log', 'inversion_logger')
    logger.info(f'Loading model.')

    training_loop(args, datasets_args, E_lr_args, D_lr_args, Dhat_lr_args,opt_args,loss_args, logger, writer,image_snapshot_ticks=2000,max_epoch=args.nepoch)


if __name__ == '__main__':
    main()
