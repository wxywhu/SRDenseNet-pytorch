
# coding: utf-8


import argparse, os
import torch
import random
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.optim as optim
from SR_DenseNet_2 import Net

from dataset import DatasetFromHdf5
import math
import torch.nn.init as init

parser = argparse.ArgumentParser(description="SR_DenseNet")
parser.add_argument("--batchSize", type=int, default=128, help="Training batch size")

parser.add_argument("--nEpochs", type=int, default=50, help="Number of epochs to train for")
parser.add_argument("--lr", type=float, default=0.0001, help="Learning Rate. Default=0.0006")
parser.add_argument("--step", type=int, default=30, help="Sets the learning rate to the initial LR decayed by momentum every n epochs, Default: n=30")
parser.add_argument("--cuda", action="store_true", help="Use cuda?")
parser.add_argument("--resume", default="", type=str, help="Path to checkpoint (default: none)")
parser.add_argument("--start-epoch", default=1, type=int, help="Manual epoch number (useful on restarts)")
parser.add_argument("--threads", type=int, default=1, help="Number of threads for data loader to use, Default: 1")
parser.add_argument('--pretrained', default='', type=str, help='path to pretrained model (default: none)')
opt = parser.parse_args()
print(opt)


cuda = opt.cuda
if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

if opt.cuda and torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

opt.seed = random.randint(1, 10000)
print("Random Seed: ", opt.seed)
torch.manual_seed(opt.seed)
if cuda:
    torch.cuda.manual_seed(opt.seed)

cudnn.benchmark = True      
print("===> Loading datasets")
train_set = DatasetFromHdf5("./train_DIV2K_96_4.h5")
training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)
print("===> Building model")

model = Net(16,16,6,6)
if torch.cuda.device_count() > 1:
  print("Let's use", torch.cuda.device_count(), "GPUs!")
  model = nn.DataParallel(model)

if torch.cuda.is_available():
   model.cuda()
   
criterion = nn.MSELoss()


print("===> Setting GPU")
if cuda:
    model = model.cuda()
    criterion = criterion.cuda()

# optionally resume from a checkpoint
if opt.resume:
    if os.path.isfile(opt.resume):
        print("=> loading checkpoint '{}'".format(opt.resume))
        checkpoint = torch.load(opt.resume)
        opt.start_epoch = checkpoint["epoch"] + 1
        model.load_state_dict(checkpoint["model"].state_dict())
    else:
        print("=> no checkpoint found at '{}'".format(opt.resume))
    
# optionally copy weights from a checkpoint
if opt.pretrained:
    if os.path.isfile(opt.pretrained):
        print("=> loading model '{}'".format(opt.pretrained))
        weights = torch.load(opt.pretrained)
        model.load_state_dict(weights['model'].state_dict())
    else:
        print("=> no model found at '{}'".format(opt.pretrained))
            
print("===> Setting Optimizer")

optimizer = optim.Adam(model.parameters(), lr=opt.lr)


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every step epochs"""
    lr = opt.lr * (0.1 ** (epoch // opt.step))
    return lr

def train(epoch):
    epoch_loss = 0
    lr = adjust_learning_rate(optimizer, epoch-1)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    print("Epoch={}, lr={}".format(epoch, optimizer.param_groups[0]["lr"]))
    
    model.train()    

    for iteration, batch in enumerate(training_data_loader, 1):
        input, target = Variable(batch[0]), Variable(batch[1], requires_grad=False)

        if opt.cuda:
            input = input.cuda()
            target = target.cuda()
        
        loss = criterion(model(input), target)
        optimizer.zero_grad()
        epoch_loss += loss.data[0]
        loss.backward() 
        #nn.utils.clip_grad_norm(model.parameters(),opt.clip)
        optimizer.step()


        if iteration%200== 0:
            print("===> Epoch[{}]({}/{}): Loss: {:.8f}".format(epoch, iteration, len(training_data_loader), loss.data[0]))
            
    print("===> Epoch {} Complete: Avg. Loss: {:.8f}".format(epoch, epoch_loss / len(training_data_loader)))


def test(epoch):
    avg_psnr = 0
    for iteration, batch in enumerate(testing_data_loader,1):
        input, target =Variable(batch[0]), Variable(batch[1])
        
        if opt.cuda:
            input =input.cuda()
            target = target.cuda()

        prediction = model(input)
        mse = criterion(prediction, target)
        psnr = 10 * math.log10(1 / mse.data[0])
        if iteration%100==0:
            print("({}/{}):psnr:{:.8f}".format(iteration, len(testing_data_loader),psnr))
            writer.add_scalar('psnr', psnr, (epoch-1)* 469 +iteration)
            
            tf.summary.scalar('psnr', psnr)
            op = tf.summary.merge_all()
            summary_str = sess.run(op)
        avg_psnr += psnr
    
    print("===> Avg. PSNR: {:.8f} dB".format(avg_psnr / len(testing_data_loader)))

def save_checkpoint(epoch):
    model_out_path = "SR_DenseNet_light_4/" + "model_{}_epoch_{}.pth".format(opt.lr,epoch)
    state = {"epoch": epoch ,"model": model}
    if not os.path.exists("SR_DenseNet_light_4/"):
        os.makedirs("SR_DenseNet_light_4/")

    torch.save(state, model_out_path)
        
    print("Checkpoint saved to {}".format(model_out_path))

print("===> Training")
for epoch in range(opt.start_epoch, opt.nEpochs + 1):
    train(epoch)
    #test(epoch)
    save_checkpoint(epoch)


