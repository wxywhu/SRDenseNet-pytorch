import argparse
import torch
import matlab.engine
import torch.nn.functional as F
from PIL import Image
from torch.autograd import Variable
import numpy as np
import time, math
import scipy.io as sio
from scipy.ndimage import gaussian_filter
from skimage.measure import compare_psnr
import os
from math import exp

parser = argparse.ArgumentParser(description="PyTorch  Test")
parser.add_argument("--cuda", action="store_true", help="use cuda?")
parser.add_argument("--model", default="model_001/model_epoch_50.pth", type=str, help="model path")
parser.add_argument("--imageset", default="Set5-output-2", type=str, help="image set")
parser.add_argument("--scale", default=2, type=int, help="scale factor, Default: 2")

def PSNR(pred, gt, shave_border=0):
    height, width = pred.shape[:2]
    pred = pred[shave_border:height - shave_border, shave_border:width - shave_border]
    gt = gt[shave_border:height - shave_border, shave_border:width - shave_border]
    imdff = pred - gt
    rmse = math.sqrt(np.mean(imdff ** 2))
    if rmse == 0:
        return 100
    return 20 * math.log10(255.0 / rmse)

def ssim(img1, img2, sd=1.5, C1=0.01**2, C2=0.03**2):
    
    mu1 = gaussian_filter(img1, sd)
    mu2 = gaussian_filter(img2, sd)
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = gaussian_filter(img1 * img1, sd) - mu1_sq
    sigma2_sq = gaussian_filter(img2 * img2, sd) - mu2_sq
    sigma12 = gaussian_filter(img1 * img2, sd) - mu1_mu2
    
    ssim_num = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2))
    ssim_den = ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    ssim_map = ssim_num / ssim_den
    mssim = np.mean(ssim_map)
    
    return mssim


opt = parser.parse_args()
cuda = opt.cuda

if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

model = torch.load(opt.model)["model"]

pathDir =  os.listdir(opt.imageset + "/")
psnr_bicubic=0.0
psnr_predicted=0.0
psnr_bicubic_average=0.0
psnr_predicted_average=0.0

ssim_bicubic = 0.0
ssim_predicted=0.0
ssim_bicubic_average=0.0
ssim_predicted_average=0.0
time_average=0.0

for name in pathDir:
    c=sio.loadmat(opt.imageset + "/" +name)['C']
    if c==3:
        im_gt_y = sio.loadmat(opt.imageset + "/" +name)['im_gt_y']
        im_b_y = sio.loadmat(opt.imageset + "/" +name)['im_b_y']
        im_l_y = sio.loadmat(opt.imageset + "/" +name)['im_l_y']
        im_b_cbcr = sio.loadmat(opt.imageset + "/" +name)['im_b_cbcr']
        #im_b = sio.loadmat(opt.imageset + "/" +name)['im_b']
    else:
        im_gt_y = sio.loadmat(opt.imageset + "/" +name)['im_gt_y']
        im_b_y = sio.loadmat(opt.imageset + "/" +name)['im_b_y']
        im_l_y = sio.loadmat(opt.imageset + "/" +name)['im_l_y']

    name = os.path.splitext(name)
    name = name[0]
    #### calculate bicubic psnr & ssim
    im_gt_y=im_gt_y.astype('float64')
    psnr_bicubic = PSNR(im_gt_y, im_b_y,shave_border=opt.scale)
    psnr_bicubic_average += psnr_bicubic
    ssim_bicubic= ssim(im_gt_y/255,im_b_y/255)
    ssim_bicubic_average += ssim_bicubic
    
    #### super resolution
    im_input = im_l_y/255.
    im_input = Variable(torch.from_numpy(im_input).float()).view(1, -1, im_input.shape[0], im_input.shape[1])
    
    if cuda:
        model = model.cuda()
        im_input = im_input.cuda()
    else:
        model = model.cpu()
    
    start_time = time.time()
    im_out= model(im_input)
    elapsed_time = time.time() - start_time
    time_average += elapsed_time

    im_out= im_out.cpu()
    out_y = im_out.data[0].numpy().astype(np.float)
    out_y *= 255.0
    out_y = out_y.clip(0., 255.)
    out_y = out_y[0,:,:]
    #### calculate result psnr & ssim
    psnr_predicted = PSNR(im_gt_y,out_y,shave_border=opt.scale)
    psnr_predicted_average +=psnr_predicted
    ssim_predicted = ssim(im_gt_y/255, out_y/255)
    ssim_predicted_average += ssim_predicted
    #### print the result
    print(name, 'predicted:', psnr_predicted,'bicubic:', psnr_bicubic)
    print(name, 'predicted:', ssim_predicted,'bicubic:', ssim_bicubic)
    



print('Scale=',opt.scale)
print('predicted_average=', psnr_predicted_average/len(pathDir), ssim_predicted_average/len(pathDir))
print('bicubic_average=', psnr_bicubic_average/len(pathDir), ssim_bicubic_average/len(pathDir))
print("It takes {}s for processing".format(time_average/len(pathDir)))


