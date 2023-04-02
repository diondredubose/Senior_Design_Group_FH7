from UNETv3Small import Unet, mobilenetv3_small
import kornia
import time
import datetime
import pandas as pd
import torch
#from sklearn.utils import shuffle
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import kornia.losses.ssim
from DepthDataset import DepthDataset, Augmentation, ToTensor
import matplotlib.pyplot as plt
import os


net = mobilenetv3_small()
model_dict = net.state_dict()
pretrained_dict = torch.load(r"/home/nano/srdsg/Senior_Design_Group_FH7/UNET/mobilenetv3-small-55df8e1f.pth")
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
model_dict.update(pretrained_dict)
net.load_state_dict(model_dict)




Model = Unet(net)



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
class GradLoss(nn.Module):
    def __init__(self):
        super(GradLoss, self).__init__()
    # L1 norm
    def forward(self, grad_fake, grad_real):
        return torch.mean( torch.abs(grad_real - grad_fake ))

def imgrad(img):
    img = torch.mean(img, 1, True)
    fx = np.array([[1 ,0 ,-1] ,[2 ,0 ,-2] ,[1 ,0 ,-1]])
    conv1 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
    weight = torch.from_numpy(fx).float().unsqueeze(0).unsqueeze(0)
    if img.is_cuda:
        weight = weight.cuda()
    conv1.weight = nn.Parameter(weight)
    grad_x = conv1(img)
    # grad y
    fy = np.array([[1 ,2 ,1] ,[0 ,0 ,0] ,[-1 ,-2 ,-1]])
    conv2 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
    weight = torch.from_numpy(fy).float().unsqueeze(0).unsqueeze(0)
    if img.is_cuda:
        weight = weight.cuda()
    conv2.weight = nn.Parameter(weight)
    grad_y = conv2(img)
    return grad_y, grad_x
def imgrad_yx(img):
    N ,C ,_ ,_ = img.size()
    grad_y, grad_x = imgrad(img)
    return torch.cat((grad_y.view(N ,C ,-1), grad_x.view(N ,C ,-1)), dim=1)





def TrainingLoop( batch_size, epochs, lr, trained_model_file_name,jetson_num):
    print("1")
    model = Model.cuda()
    print("2")
    LOAD_DIR = "/home/srdsg/Senior_Design_Group_FH7/UNET"
    print("3")

    folder_name = "jetson_{}".format(jetson_num)
    current_dir = os.getcwd()
    folder_path = os.path.join(current_dir, folder_name)

    if os.path.exists(folder_path):
        os.chdir(folder_path)
        print(f"Changed working directory to {folder_path}")
    else:
        print(f"Directory {folder_path} does not exist")

    model.load_state_dict(torch.load('UNET_MBIMAGENET.pth'.format(LOAD_DIR,jetson_num)))

    current_dir = os.getcwd()
    if os.path.basename(current_dir) == "jetson_{}".format(jetson_num):
        os.chdir('..')
        print(f"Exited {current_dir}, current working directory is now {os.getcwd()}")
    else:
        print('Error: current directory is not jetson_{}'.format(jetson_num))

    print("Model Loaded.")

    # epochs = 50
    # lr = 0.0001
    # batch_size = 2

    traincsv = pd.read_csv(r"/home/nano/srdsg/nyu_data/data/nyu2_train.csv")
    traincsv = traincsv.values.tolist()
    # traincsv = shuffle(traincsv, random_state=2)

    depth_dataset = DepthDataset(traincsv=traincsv, root_dir=r"/home/nano/srdsg/nyu_data/",
                                 transform=transforms.Compose([Augmentation(probability=.6),
                                                               ToTensor()]))  # can add augmentation when u have small datasets
    depth_dataset, test_dataset = torch.utils.data.random_split(depth_dataset, [10, len(depth_dataset) - 10])
    train_loader = DataLoader(depth_dataset, batch_size, shuffle=True)

    #print(len(train_loader))
    l1_criterion = GradLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr, amsgrad=True)
    loss_list = []
    epoch_list = []

    for epoch in range(epochs):
        print("4")
        #torch.save(model.state_dict(), '{}/{}'.format(LOAD_DIR,"UNET_MBIMAGENET.pth"))
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

            # output = DepthNorm(output)
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
        # loss_list.append(losses.avg)
        # epoch_list.append(epoch)
        # plt.plot(epoch_list, loss_list)
        # plt.xlabel("Epoch")
        # plt.ylabel("Loss")
        # plt.show()
        print("5")
        torch.save(model.state_dict(), '{}'.format(trained_model_file_name))
        print("6")
    return model.state_dict()
