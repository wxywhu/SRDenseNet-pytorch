# SRDenseNet-pytorch
Implementation of paper: "Image Super-Resolution Using Dense Skip Connections"(http://openaccess.thecvf.com/content_ICCV_2017/papers/Tong_Image_Super-Resolution_Using_ICCV_2017_paper.pdf) in PyTorch

## Usage
### Training
```
usage: main.py [-h] [--batchSize BATCHSIZE] [--nEpochs NEPOCHS] [--lr LR]
               [--step STEP] [--cuda] [--resume RESUME]
               [--start-epoch START_EPOCH] [--threads THREADS]
               [--pretrained PRETRAINED]

Pytorch SRDenseNet train

optional arguments:
  -h, --help            show this help message and exit
  --batchSize BATCHSIZE
                        training batch size
  --nEpochs NEPOCHS     number of epochs to train for
  --lr LR               Learning Rate. Default=1e-4
  --step STEP           Sets the learning rate to the initial LR decayed by
                        momentum every n epochs, Default: n=30
  --cuda                Use cuda?
  --resume RESUME       Path to checkpoint (default: none)
  --start-epoch START_EPOCH
                        Manual epoch number (useful on restarts)
  --threads THREADS     Number of threads for data loader to use, Default: 1
  --pretrained PRETRAINED
                        path to pretrained model (default: none)

```
### Test
```
usage: test.py [-h] [--cuda] [--model MODEL] [--imageset IMAGESET] [--scale SCALE]

Pytorch SRDenseNet Test

optional arguments:
  -h, --help     show this help message and exit
  --cuda         use cuda?
  --model MODEL  model path
  --imageset IMAGESET  imageset name
  --scale SCALE  scale factor, Default: 4
```

### Prepare Training dataset
 The training data is generated with Matlab Bicubic Interplotation, please refer [Code for Data Generation](https://github.com/wxywhu/SRDenseNet-pytorch/tree/master/data) for creating training files.

### Prepare Test dataset
 The test imageset is generated with Matlab Bicubic Interplotation, please refer [Code for test](https://github.com/wxywhu/SRDenseNet-pytorch/tree/master/TestSet) for creating test imageset.
 
### Performance
 We provide a pretrained SRDenseNet x4 model trained on DIV2K images from [DIV2K_train_HR] (http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip)   
 Non-overlapping sub-images with a size of 96 × 96 were cropped in the HR space.
 Other settings is the same as the original paper
 
 - Performance in PSNR on Set5, Set14, and BSD100
  
| DataSet/Method        | LapSRN Paper          | LapSRN PyTorch|
| ------------- |:----------:|:----------:|
| Set5      | 32.02/0.893      | **31.57/0.883** |
| Set14     | 28.50/0.778      | **28.11/0.771** |
| BSD100    | 27.53/0.733      | **27.32/0.729** |
