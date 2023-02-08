
import numpy as np
import os
import time
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models.resnet import resnet101
import torch.utils.data as data
from PIL import Image
from torchvision.transforms import Resize, Compose, ToPILImage, ToTensor
import pickle
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


import torch.nn.functional as f
import segmentation_models_pytorch as smp

def rgb2gray(rgb):
    return np.dot(rgb[... ,:3], [0.299, 0.587, 0.144])

UNET = smp.Unet(
    encoder_name="mobilenet_v2",
    encoder_weights = "imagenet",
    in_channels = 3,
    classes = 1
)



DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOAD_DIR = "."


class ECJDataset(data.Dataset):
    def __init__(self, train=True):
        self.load_type = "train" if train else "test"
        if train:
            self.name_map = pickle.load(open("{}/DataSet/index1.pkl".format(LOAD_DIR), 'rb'))
            self.rgb_paths = list(self.name_map.keys())
        else:
            self.name_map = pickle.load(open("{}/DataSet/index2.pkl".format(LOAD_DIR), 'rb'))
            self.rgb_paths = list(self.name_map.keys())
        self.rgb_transform = Compose(
            [ToTensor()])  # ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        self.depth_transform = Compose([ToTensor()])
        self.length = len(self.rgb_paths)

    def __getitem__(self, index):
        path = '{}/DataSet/{}_Images/'.format(LOAD_DIR, self.load_type) + self.rgb_paths[index]
        rgb = Image.open(path)
        img_rgb = mpimg.imread(path)
        img_gray = rgb2gray(img_rgb)
        depth = Image.open \
            ('{}/DataSet/{}_Depth/'.format(LOAD_DIR, self.load_type) + self.name_map[self.rgb_paths[index]])
        depth = depth.resize((576,576))
        plt.imshow(img_gray, interpolation='nearest')
        plt.show()
        transform = transforms.Pad(2)
        rgb = transform(rgb)
        return self.rgb_transform(rgb).float(), self.depth_transform(depth).float()

    def __len__(self):
        return self.length

class GradLoss(nn.Module):
    def __init__(self):
        super(GradLoss, self).__init__()
    # L1 norm
    def forward(self, grad_fake, grad_real):
        return torch.mean( torch.abs(grad_real -grad_fake) )

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



if __name__ == '__main__':
    # hyperparams

    lr = 0.001
    bs = 4
    # dataset
    train_dataset = ECJDataset()
    train_size = len(train_dataset)
    eval_dataset = ECJDataset(train=False)
    print(train_size)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=bs ,shuffle=True)
    eval_dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=bs ,shuffle=True)

    first_data = train_dataset[0]

    features, set = first_data

    # print(features, set)

    # network initialization
    UNET = UNET.to(DEVICE)
    try:
        UNET.load_state_dict(torch.load('{}/fyn_model.pt'.format(LOAD_DIR)))
        print("loaded model from drive")
    except:
        print('Initializing model...')
        print('Done!')
    # optimizer
    optimizer = torch.optim.Adam(UNET.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=4e-5)
    # loss function
    grad_criterion = GradLoss()
    # start training
    for epoch in range(0, 5):
        try:
            torch.cuda.empty_cache()
        except:
            pass
        UNET.train()
        start = time.time()
        # learning rate decay
        if epoch > 5:
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * 0.9
        for i ,(data ,z) in enumerate(train_dataloader):
            data ,z = Variable(data.to(DEVICE)) ,Variable(z.to(DEVICE))
            optimizer.zero_grad()
            z_fake = UNET(data)
            grad_real, grad_fake = imgrad_yx(z), imgrad_yx(z_fake)
            loss = grad_criterion(grad_fake, grad_real)
            loss.backward()
            optimizer.step()
            if ( i +1) % 5 == 0:
                print("[epoch %2d][iter %4d] loss: %.4f" % (epoch, i, loss))
        # save model
        torch.save(UNET.state_dict() ,'{}/fyn_model.pt'.format(LOAD_DIR))
        end = time.time()
        print('model saved')
        print('time elapsed: %fs' % (end - start))
        # Evaluation
        if (epoch +1) % 5 == 0:
            try:
                torch.cuda.empty_cache()
            except:
                pass
            UNET.eval()
            print('evaluating...')
            eval_loss = 0
            count = 0
            with torch.no_grad():
              for i ,(data ,z) in enumerate(eval_dataloader):
                  data ,z = Variable(data.to(DEVICE)) ,Variable(z.to(DEVICE))
                  z_fake = UNET(data)
                  eval_loss += float(data.size(0)) * grad_criterion(z_fake, z).item()**2
                  count += float(data.size(0))
            print("[epoch %2d] RMSE_log: %.4f" % (epoch, math.sqrt(eval_loss /count)))
