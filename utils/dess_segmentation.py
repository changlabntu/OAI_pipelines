import torch.nn as nn
from torch.nn import functional as F
import torch
from torchvision import models
import glob, os
import numpy as np
from skimage import measure
from PIL import Image
import pandas as pd


def to_8bit(x):
    """
    turn float images to 8bit images for display
    """
    x = x - x.min()
    x = (x / x.max() * 255).astype(np.uint8)

    if len(x.shape) == 2:
        x = np.concatenate([np.expand_dims(x, 2)]*3, 2)
    return x


def imagesc(x, show=True, save=None):
    """
    display images
    """
    if isinstance(x, list):
        x = [to_8bit(y) for y in x]
        x = np.concatenate(x, 1)
        x = Image.fromarray(x)
    else:
        x = x - x.min()
        x = Image.fromarray(to_8bit(x))
    if show:
        x.show()
    if save:
        x.save(save)
    return x


class conv_block(nn.Module):
    """
    Convolution Block
    """

    def __init__(self, in_ch, out_ch, coord=False):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    """
    Up Convolution Block
    """

    def __init__(self, in_ch, out_ch, coord=False):
        super(up_conv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True)
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            self.conv,
            #nn.ConvTranspose2d(out_ch, out_ch, 2, 2),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


def get_encoder(self, name, depth):
    backbone = getattr(models, name)(pretrained=True)

    if name in ['resnet34', 'resnet50', 'resnet101']:
        resnet_dict = {
            'resnet34': [64, 64, 128, 256, 512],
            'resnet50': [64, 256, 512, 1024, 2048],
            'resnet101': [64, 256, 512, 1024, 2048],
        }
        enco_fil = resnet_dict[name]
        setattr(self, 'encoder0', nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu))
        setattr(self, 'encoder1', nn.Sequential(backbone.maxpool, backbone.layer1))
        for i in range(2, depth):
            setattr(self, 'encoder' + str(i), getattr(backbone, 'layer' + str(i)))

    if name in ['vgg11', 'vgg11_bn', 'vgg16', 'vgg16_bn', 'vgg19']:
        enco_fil = [64, 128, 256, 512, 512]
        vgg_dict = {
            'vgg11': [None, 2, 5, 10, 15, 20],
            'vgg11_bn': [None, 3, 7, 14, 21, 28],
            'vgg16': [None, 4, 9, 16, 23, 30],
            'vgg16_bn': [None, 6, 13, 23, 33, 43],
            'vgg19': [None, 4, 9, 18, 27, 36]
        }
        loc = vgg_dict[name]

        for i in range(depth):
            setattr(self, 'encoder' + str(i), backbone.features[loc[i]:loc[i + 1]])

    return enco_fil


class UNet_clean(nn.Module):
    """
    Attention Unet implementation
    Paper: https://arxiv.org/abs/1804.03999
    """

    def __init__(self, output_ch=1, backbone='vgg11', depth=5):
        super(UNet_clean, self).__init__()

        self.depth = depth
        self.skip = [1, 1, 1, 1]  # (len = depth-1)
        deco_fil = [x * 32 for x in [1, 2, 4, 8, 8]]

        self.coord = False
        enco_fil = get_encoder(self, backbone, depth)
        skip_fil = enco_fil

        # Up Path
        for i in range(depth - 3, -1, -1):  # [2, 1, 0]
            setattr(self, 'Up'+str(i), up_conv(deco_fil[i+1], deco_fil[i], coord=self.coord))
        # First up layer
        i = depth - 2
        setattr(self, 'Up'+str(i), up_conv(enco_fil[i+1], deco_fil[i], coord=self.coord))

        for i in range(depth - 2, -1, -1):  # [3, 2, 1, 0]
            setattr(self, 'Up_conv'+str(i),
                    conv_block(skip_fil[i] * self.skip[i] + deco_fil[i], deco_fil[i], coord=self.coord))

        # Final
        self.Conv = nn.Conv2d(deco_fil[0], output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # x (b, t, c, w, h)
        B = x.shape[0]
        depth = self.depth
        e = [self.encoder0(x)]
        for i in range(1, depth):
            e.append(getattr(self, 'encoder'+str(i))(e[i-1]))  # (b*t, c, w, h)

        # Skip connection path
        for i in range(depth):
            e[i] = e[i]

        # Decoder path
        d = e[-1]
        for i in range(depth - 2, -1, -1):
            d = getattr(self, 'Up'+str(i))(d)
            if self.skip[i]:
                d = torch.cat((e[i], d), dim=1)
            d = getattr(self, 'Up_conv'+str(i))(d)
        d = self.Conv(d)
        return d,


def gather_npy(d):
    npys = glob.glob(d + '/*.npy')
    npys.sort()
    x = []
    for npy in npys:
        s = np.load(npy)
        s = np.expand_dims(s, 0)
        x.append(s)
    x = np.concatenate(x, 0)
    return x


def dess_to_segmentation(net, dir_list, destination, sequence=None):
    for d in dir_list:
        ID = d.split('/')[-1]

        x = gather_npy(d)
        print(x.shape)
        thickness = x.shape[0]

        if sequence == 'tse':
            x = x[:, :416, :416]

        x = np.concatenate([np.expand_dims(x, 1)] * 3, 1)
        x[x >= 400] = 400
        x = x / x.max()
        x = torch.from_numpy(x.astype(np.float32))
        x = x.cuda()

        y0, = net(x[:(thickness//2), ::])
        y0 = y0.cpu().detach().numpy()
        y1, = net(x[(thickness//2):, ::])
        y1 = y1.cpu().detach().numpy()
        y = np.concatenate([y0, y1], 0)
        y = np.argmax(y, 1)
        y = y.astype(np.uint8)

        if not os.path.isdir(destination):
            os.mkdir(destination)
        np.save(destination + ID + '.npy', y)


def clean_bone_blobs(dir_list, destination):
    for d in dir_list:
        x = np.load(d)
        bone = ((x == 1) + (x == 2))
        bone_blobs = measure.label(bone)
        bone_blobs_size = []
        for i in range(bone_blobs.max()):
            bone_blobs_size.append((bone_blobs == i).sum())
        bone_blobs_size = np.array(bone_blobs_size)
        background_index = np.argsort(bone_blobs_size)[-1]
        femur_index = np.argsort(bone_blobs_size)[-2]
        tibia_index = np.argsort(bone_blobs_size)[-3]
        background = (bone_blobs == background_index)
        femur = (bone_blobs == femur_index)
        tibia = (bone_blobs == tibia_index)
        print(id)
        ratio = (femur.sum()+tibia.sum()) / bone.sum()
        print('Extracted bone percentage = ' + str(ratio))
        #if ratio <= 0.9:
        #    return 0

        y = 0 * x
        y[femur == 1] = 1
        y[tibia == 1] = 2
        y[x == 3] = 3
        y[x == 4] = 4

        if not os.path.isdir(destination):
            os.mkdir(destination)
        ID = d.split('/')[-1]
        np.save(destination + ID, y)


def try_histogram_matching():
    from skimage.exposure import match_histograms

    lt = glob.glob('/media/ghc/GHc_data1/OAI_extracted/OAI00womac3Npy/SAG_IW_TSE_LEFT/*')
    ld = glob.glob('/media/ghc/GHc_data1/OAI_extracted/OAI00womac3Npy/SAG_3D_DESS_LEFT/*')
    lt.sort()
    ld.sort()

    t = gather_npy(lt[0])
    d = gather_npy(ld[0])

    #tm = match_histograms(t, d, multichannel=False)
    x = torch.from_numpy(t.astype(np.float32))
    x = torch.cat([x.unsqueeze(1)] * 3, 1)
    x[x >= 800] = 800
    x = x / x.max()
    x.cuda()
    y, = net(x[:, :, :432, :432])
    y = y.cpu().detach().numpy()


def filter_by_blobs(blobs, blobs_to_select):
    blobs_size = []
    for i in range(blobs.max()):
        blobs_size.append((blobs == i).sum())
    blobs_size = np.array(blobs_size)

    indices = [np.argsort(blobs_size)[x] for x in blobs_to_select]
    selected = 0 * blobs
    for index in indices:
        selected = selected + (blobs == index)

    side = selected.sum(0)
    front = selected.sum(1)
    sidebox = [np.nonzero(side.sum(1))[0][0], np.nonzero(side.sum(1))[0][-1], np.nonzero(side.sum(0))[0][0], np.nonzero(side.sum(0))[0][-1]]
    center0 = (sidebox[0] + sidebox[1]) // 2
    center1 = (sidebox[2] + sidebox[3]) // 2
    frontbox = [np.nonzero(front.sum(1))[0][0], np.nonzero(front.sum(1))[0][-1] + 1]

    coor = [frontbox[0], frontbox[1], center0 - 112, center0 + 112, center1 - 112, center1 + 112]
    selected = selected[coor[0]:coor[1], coor[2]:coor[3], coor[4]:coor[5]]
    return selected, coor


def crop_by_cartilage(npy, destination):
    y = np.load(npy)
    blobs = measure.label(y == 3)

    selected, coor = filter_by_blobs(blobs, blobs_to_select=[-2])
    selected_thickness = selected.shape[0]
    if selected_thickness < 20:
        selected, coor = filter_by_blobs(blobs, blobs_to_select=[-2, -3])
        selected_thickness = selected.shape[0]

    sideview = selected.sum(0)
    sideview = sideview / sideview.max()
    frontview = torch.nn.Upsample(size=(224, 224))(torch.from_numpy(selected.sum(1)).unsqueeze(0).unsqueeze(1).type(torch.float32)).squeeze().numpy()
    frontview = frontview / frontview.max()

    ID = npy.split('/')[-1].split('.')[0]
    if not os.path.isdir(destination):
        os.mkdir(destination)
    imagesc(np.concatenate([sideview, frontview], 1), show=False, save=destination + ID + '_' + str(selected_thickness) + '.jpg')
    print(npy)
    return npy, coor


if 0:
    #  network
    net = UNet_clean(output_ch=4, backbone='vgg11', depth=5)
    net.load_state_dict(torch.load('utils/clean_femur_tibia_cartilage.pth'))
    net = net.cuda()
    net = nn.DataParallel(net)

    # dess_to_segmentation
    dess_to_segmentation(net,
                         dir_list=glob.glob('/media/ghc/GHc_data1/OAI_extracted/OAI00womac3Npy/SAG_IW_TSE_RIGHT/*'),
                         destination='/media/ghc/GHc_data1/OAI_segmented/SAG_IW_TSE_RIGHT/',
                         sequence='tse')

if 0:
    clean_bone_blobs(dir_list=glob.glob('/media/ghc/GHc_data1/OAI_segmented/SAG_3D_DESS_RIGHT/*'),
                     destination='/media/ghc/GHc_data1/OAI_segmented/SAG_3D_DESS_RIGHT_clean/')

#for npy in npy_list:
#    crop_by_cartilage(npy, destination='/media/ghc/GHc_data1/OAI_segmented/SAG_IW_TSE_RIGHT_crop/')

#npy_list = sorted(glob.glob('/Users/ghc/Dropbox/Z_DL/temp_share/SAG_IW_TSE_RIGHT/*'))
npy_list = glob.glob('/media/ghc/GHc_data1/OAI_segmented/SAG_IW_TSE_RIGHT/*')
destination = '/media/ghc/GHc_data1/OAI_segmented/SAG_IW_TSE_RIGHT_crop/'
df = list(map(crop_by_cartilage, npy_list, [destination] * len(npy_list)))
pd.DataFrame(df, columns=['npy', 'coor']).to_csv(destination + 'crop_coor.csv', index=False)
