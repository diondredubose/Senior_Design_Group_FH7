
import numpy as np
import time
import torch
from torch.autograd import Variable
import torch.nn as nn
from torchvision import transforms
import torch.utils.data as data
from PIL import Image
from torchvision.transforms import Resize, Compose, ToPILImage, ToTensor
import pickle
import math
import matplotlib.image as mpimg
import segmentation_models_pytorch as smp # Not sure if this library is compatible with JETSON
import gc
import pickle

from JustTest import VNL_Loss, RMSE_log, NormalLoss, GradLoss, SiLogLoss, MseIgnoreZeroLoss

gc.collect()

torch.cuda.empty_cache()
def rgb2gray(rgb):
    return np.dot(rgb[... ,:3], [0.299, 0.587, 0.144])

############################
############################
#MODEL

#AVAILABLE ENCODERS
#timm-mobilenetv3_large_075
#timm-mobilenetv3_large_100
#timm-mobilenetv3_large_minimal_100
#timm-mobilenetv3_small_075
#timm-mobilenetv3_small_100
#timm-mobilenetv3_small_minimal_100

UNET = smp.Unet(
    encoder_name="timm-mobilenetv3_small_075",
    encoder_weights = "imagenet",

    in_channels = 3, #Default 3 since we are inputting RGB images
    classes = 1 #1 output
)

############################
############################

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOAD_DIR = "."

############################
############################
#DATASET

class ECJDataset(data.Dataset):
    def __init__(self, train=True):
        self.load_type = "train" if train else "test"
        if train:
            self.name_map = pickle.load(open("{}/data/nyu2_train1.pkl".format(LOAD_DIR), 'rb'))
            self.rgb_paths = list(self.name_map.keys())
        else:
            self.name_map = pickle.load(open("{}/data/nyu2_test1.pkl".format(LOAD_DIR), 'rb'))
            self.rgb_paths = list(self.name_map.keys())
        self.rgb_transform = Compose(
            [ToTensor()])  # ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        self.depth_transform = Compose([ToTensor()])
        self.length = len(self.rgb_paths)

    def __getitem__(self, index):
        path = self.rgb_paths[index]
        rgb = Image.open(path)
        depth = Image.open(self.name_map[self.rgb_paths[index]])

        return self.rgb_transform(rgb).float(), self.depth_transform(depth).float()

    def __len__(self):
        return self.length

############################
############################
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

Output = r"C:\Users\Admin\Downloads\pytorch_ipynb\data"
DataLoss = []

class MaskedL1Loss(nn.Module):
    def __init__(self):
        super(MaskedL1Loss, self).__init__()

    def forward(self, pred, target):
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        valid_mask = (target > 0).detach()
        diff = target - pred
        diff = diff[valid_mask]
        self.loss = diff.abs().mean()
        return self.loss



if __name__ == '__main__':

    depth_criterion = RMSE_log()
    normal_criterion  = MaskedL1Loss()
    grad_criterion = GradLoss()
    #normal_criterion = NormalLoss()
    grad_factor = 10
    normal_factor = 1
    open("{}/Loss.pkl".format(Output), 'w').close()



    lr = 0.001

    bs = 6
    # dataset
    train_dataset = ECJDataset()
    train_size = len(train_dataset)
    eval_dataset = ECJDataset(train=False)
    test_size = len(eval_dataset)
    print(train_size)


    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=bs,shuffle=True)
    eval_dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=bs,shuffle=True)
    # network initialization
    rgb, depth = train_dataset[0]
    UNET = UNET.to(DEVICE)
    try:
        UNET.load_state_dict(torch.load('{}/UNET_NYUMIX_model.pt'.format(LOAD_DIR)))
        print("loaded model from drive")
    except:
        print('Initializing model...')
        print('Done!')
    # optimizer
    optimizer = torch.optim.Adam(UNET.parameters(), lr=lr, amsgrad = True, weight_decay=1e-4)
    # loss function
    grad_criterion = GradLoss()
    # start training
    for epoch in range(0, 10):
        try:
            torch.cuda.empty_cache()
        except:
            pass
        UNET.train()
        start = time.time()
        # learning rate decay
        if epoch > 5 :
             for param_group in optimizer.param_groups:
                 param_group['lr'] = param_group['lr'] * 0.9
        for i,(data,z) in enumerate(train_dataloader):
            data,z = Variable(data.to(DEVICE)),Variable(z.to(DEVICE))
            optimizer.zero_grad()
            z_fake = UNET(data)

            depth_loss = depth_criterion(z_fake, z)
            grad_real, grad_fake = imgrad_yx(z), imgrad_yx(z_fake)
            grad_loss = grad_criterion(grad_fake, grad_real) * 10
            normal_loss = normal_criterion(grad_fake, grad_real) * normal_factor

            loss = normal_loss
            loss.backward()
            optimizer.step()
            if (i+1) % 10 == 0:
                DataLoss += [normal_loss]
                print("[epoch %2d][iter %4d] loss: %.4f  N:%.4f" % (epoch, i, depth_loss.mean(),  normal_loss))
        # save model
        torch.save(UNET.state_dict(),'{}/UNET_NYUMIX_model.pt'.format(LOAD_DIR))
        end = time.time()
        print('model saved')
        print('time elapsed: %fs' % (end - start))
        if(epoch == 5):
            pickle_out = open(r"C:\Users\Admin\Downloads\pytorch_ipynb\DataSet\Loss.pkl", "wb")
            pickle.dump(DataLoss, pickle_out)
            pickle_out.close()
            print(DataLoss)

        if (epoch+1) % 1 == 0:
            try:
                torch.cuda.empty_cache()
            except:
                pass
            UNET.eval()
            print('evaluating...')
            eval_loss = 0
            count = 0
            with torch.no_grad():
              for i,(data,z) in enumerate(eval_dataloader):
                  data,z = Variable(data.to(DEVICE)),Variable(z.to(DEVICE))
                  z_fake = UNET(data)

                  depth_loss = depth_criterion(z_fake, z)
                  grad_real, grad_fake = imgrad_yx(z), imgrad_yx(z_fake)
                  grad_loss = grad_criterion(grad_fake, grad_real) * grad_factor
                  normal_loss = normal_criterion(grad_fake, grad_real) * normal_factor

                  loss = depth_loss + grad_loss + normal_loss
                  eval_loss += float(data.size(0)) * loss.mean().item()**2
                  count += float(data.size(0))
            print("[epoch %2d] RMSE_log: %.4f" % (epoch, math.sqrt(eval_loss/count)))