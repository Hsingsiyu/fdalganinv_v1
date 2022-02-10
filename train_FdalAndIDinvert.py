# --coding:utf-8--
#from datasets.celebahq import Config
import os
os.chdir('/home/xsy/idinvert_pytorch-mycode/') # convenient for debug
import argparse
from datetime import datetime
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from training.misc import EasyDict
from training.train_loop import training_loop
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
    parser.add_argument('--train_batch_size', type=int, default=4,
                        help='training batch size')
    parser.add_argument('--test_batch_size', type=int, default=4,
                        help='training batch size')
    parser.add_argument('--cuda', type=bool, default=True,
                        help='to use cuda or not')
    parser.add_argument('--gpu_ids', type=list, default=None, #[0,1,2,3],
                        help='list of gpus')
    parser.add_argument('--test_save_step', type=int, default=0,
                        help='how much step to be saved when inference')
    parser.add_argument('--save_root', type=str, default='/home/xsy/idinvert_pytorch-mycode/fdaloutput/')
    parser.add_argument('--divergence', type=str, default='kl')
    parser.add_argument('--reg_coef', type=int, default=1)
    parser.add_argument('--nepoch', type=int, default=5000)
    parser.add_argument('--vgg', type=bool, default=False)
    parser.add_argument('--learn_rate', type=int, default=0.0001)
    args = parser.parse_args()

    current_time = datetime.now().strftime('%b%d_%H-%M')
    prefix = 'fDAL-FFHQ-InitEncoder_StyleD'
    parm='_train_lr%s_bs_%s_epoch%s_regcoef%s_vgg%s'%(args.learn_rate,args.train_batch_size,args.nepoch,args.reg_coef,args.vgg)
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

    class Config:
        data_root = args.data_root
        size = args.image_size
        min_val = -1.0
        max_val = 1.0
        split=1000 #65000
    datasets_args = Config()

    opt_args = EasyDict(betas=(0.9, 0.99), eps=1e-8)
    E_lr_args = EasyDict(learning_rate=args.learn_rate, decay_step=300, decay_rate=0.8, stair=False)
    D_lr_args = EasyDict(learning_rate=0.00001, decay_step=3000, decay_rate=0.8, stair=False)


    logger = setup_logger(args.save_logs, 'inversion.log', 'inversion_logger')
    logger.info(f'Loading model.')

    training_loop(args, datasets_args, E_lr_args, D_lr_args, opt_args, logger, writer,image_snapshot_ticks=50,max_epoch=args.nepoch)


if __name__ == '__main__':
    main()
