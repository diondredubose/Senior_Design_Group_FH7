from UNETV3 import Unet, mobilenetv3_large
import kornia
import time
import datetime
import pandas as pd
import torch
from sklearn.utils import shuffle
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import kornia.losses.ssim
from DepthDataset import DepthDataset, Augmentation, ToTensor
import matplotlib.pyplot as plt




Model = Unet(mobilenetv3_large())

def SSIM(img1, img2, val_range, window_size=11, window=None, size_average=True, full=False):
    return kornia.losses.ssim_loss(img1, img2,window_size=11, max_val=val_range, reduction='none')





def DepthNorm(depth, maxDepth=1000.0):
    return maxDepth / depth


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



# from data import getTrainingTestingData
# from utils import AverageMeter, DepthNorm, colorize

model = Model.cuda()
LOAD_DIR = "."
model.load_state_dict(torch.load('{}/UNET.pth'.format(LOAD_DIR)))
print('Model Loaded.')

epochs = 50
lr = 0.0001
batch_size = 10

traincsv=pd.read_csv(r"C:\Users\Admin\Downloads\pytorch_ipynb\data\nyu2_train.csv")
traincsv = traincsv.values.tolist()
traincsv = shuffle(traincsv, random_state=2)



depth_dataset = DepthDataset(traincsv=traincsv, root_dir=r"C:\Users\Admin\Downloads\pytorch_ipynb",
                             transform=transforms.Compose([ ToTensor()]))#can add augmentation when u have small datasets
train_loader = DataLoader(depth_dataset, batch_size, shuffle=True)
l1_criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr, amsgrad= True)
loss_list = []
epoch_list = []
# Start training...
for epoch in range(epochs):

    torch.save(model.state_dict(), '{}/UNET.pth'.format(LOAD_DIR))
    batch_time = AverageMeter()
    losses = AverageMeter()
    N = len(train_loader)

    # Switch to train mode
    model.train()

    end = time.time()

    for i, sample_batched in enumerate(train_loader):
        optimizer.zero_grad()

        # Prepare sample and target
        image = torch.autograd.Variable(sample_batched['image'].cuda())
        depth = torch.autograd.Variable(sample_batched['depth'].cuda(non_blocking=True))

        # Normalize depth
        depth_n = DepthNorm(depth)

        # Predict
        output = model(image)



        #output = DepthNorm(output)
        # Compute the loss
        l_depth = l1_criterion(output, depth_n)
        l_ssim = torch.clamp((1 - SSIM(output, depth_n, val_range=1000.0 / 10.0)) * 0.5, 0, 1)

        loss = (1.0 * l_ssim.mean().item()) + (0.1 * l_depth)

        # Update step

        losses.update(loss.data.item(), image.size(0))
        loss.backward()
        optimizer.step()

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        eta = str(datetime.timedelta(seconds=int(batch_time.val * (N - i))))

        # Log progress
        niter = epoch * N + i

        if i % 5 == 0:
            # Print to console
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.sum:.3f})\t'
                  'ETA {eta}\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})'
                  .format(epoch, i, N, batch_time=batch_time, loss=losses, eta=eta))

            # Log to tensorboard
            # writer.add_scalar('Train/Loss', losses.val, niter)
    loss_list.append(losses.avg)
    epoch_list.append(epoch)
    plt.plot(epoch_list, loss_list)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()
    torch.save(model.state_dict(), '{}/UNET.pth'.format(LOAD_DIR))
