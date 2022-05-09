import os
import glob

from PIL import Image
import torch
from torchvision import datasets,transforms
from swd import swd

import lpips
from piq import ssim,psnr
# # PSNR
# from piqa import PSNR
# # SSIM
# from piqa import SSIM

torch.manual_seed(123)
#fix seed
path1='/home/xsy/idinvert_pytorch-mycode/results/0427_quan_val/src'
path2=''
batch_size=900
data_transforms=transforms.Compose([
    transforms.ToTensor(),
# transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
img1=datasets.ImageFolder(path1,transform=data_transforms)
imgLoader_1=torch.utils.data.DataLoader(img1,batch_size=batch_size)

# img2=datasets.ImageFolder(path2,transform=None)
# imgLoader_2=torch.utils.data.Dataloader(img2,batch_size=batch_size)



input1,labels=next(iter(imgLoader_1))#[BN,3,W,H]
# print(input1.shape)
# print(labels)
# print(labels.shape)

origin_img=input1[:300,]
baseline_img=input1[300:600,]
our_img=input1[600:900,]

ssim_out=ssim(origin_img,baseline_img)
print(f'ssim baseline:{ssim_out.item():.3f}')

ssim_out=ssim(origin_img,our_img)
print(f'ssim our:{ssim_out.item():.3f}')

psnr_out=psnr(origin_img,baseline_img)
print(f'psnr baseline:{psnr_out.item():.3f}')

psnr_out=psnr(origin_img,our_img)
print(f'psnr our:{psnr_out.item():.3f}')
# print(input1)
# input2,_=next(iter(imgLoader_2))#[BN,3,W,H]
#
# ssim = SSIM()
# l =ssim(input1, input2)
# print("ssim::%f"(l))
# psnr = PSNR()
# l = psnr(input1, input2)
# print("ssim::%f"(l))

# swd_out=swd(origin_img,baseline_img,device="cuda")
# print("swd baseline:%f"%(swd_out.item()))
#
# swd_out=swd(origin_img,our_img,device="cuda")
# print("swd our:%f"%(swd_out.item()))
# #
# MSE_out=torch.mean((origin_img-baseline_img)**2)
# print("MSE_out baseline:")
# print(MSE_out.item())
#
# MSE_out=torch.mean((origin_img-our_img)**2)
# print("MSE_out ours:")
# print(MSE_out.item())
# #
# # # FID,SWD
# # # MSE,LPIPS,
# # # SSIM,PSNR
#
# loss_fn_alex=lpips.LPIPS(net='alex')
# loss_fn_vgg=lpips.LPIPS(net='vgg')
#
# with torch.no_grad():
#     d_alex = loss_fn_alex((origin_img), (baseline_img))
#     d_vgg=loss_fn_vgg((origin_img), (baseline_img))
# print(f"LPIPS baseline: {d_alex.mean().item():.3f} {d_vgg.mean().item():.3f}")
#
# with torch.no_grad():
#     d_alex = loss_fn_alex((origin_img), (our_img))
#     d_vgg=loss_fn_vgg((origin_img), (our_img))
# print(f"LPIPS our: {d_alex.mean().item():.3f} {d_vgg.mean().item():.3f}")

