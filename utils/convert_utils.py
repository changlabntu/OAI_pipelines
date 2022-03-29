import numpy as np
import os, glob
from utils.images_utils import imagesc
import pandas as pd
import tifffile as tiff


def print_summary():
    ls = sorted(glob.glob('/media/ghc/GHc_data2/OAI_extracted/womac5min0/Processed/TSE/*LEFT.npy'))
    for l in ls:
        name = l.split('/')[-1].split('.')[0]
        left = np.load(l)
        right = np.load(l.replace('LEFT', 'RIGHT'))
        left = np.concatenate([left[:, :, i] for i in range(left.shape[2])], 1)
        right = np.concatenate([right[:, :, i] for i in range(right.shape[2])], 1)
        all = np.concatenate([left, right], 0)
        imagesc(all, show=False,
                save='/media/ghc/GHc_data2/OAI_extracted/womac5min0/Processed/summary/'+name+'.png')


root = '/media/ghc/GHc_data2/OAI_extracted/womac5min0/Processed/'
source = root + 'TSE/'
destination = root + 'full/'
os.makedirs(destination, exist_ok=True)
os.makedirs(destination + 'a/', exist_ok=True)
os.makedirs(destination + 'b/', exist_ok=True)

df = pd.read_csv('/media/ghc/GHc_data2/OAI_extracted/womac5min0/womac5min0.csv')
df['labels'] = [(x[-1] == 'R') for x in df[df.columns[6:]].idxmax(axis=1)]

for i in range(df.shape[0])[:]:
    name = str(df.iloc[i]['ID']) + '_' + str(df.iloc[i]['VER']).zfill(2) + '_' + str(df.iloc[i]['SIDE'])
    label = df.iloc[i]['labels']
    side = df.iloc[i]['SIDE']
    npy = np.load(source + name + '.npy')
    savename = str(df.iloc[i]['ID']) + '_' + str(df.iloc[i]['VER']).zfill(2)
    for s in range(npy.shape[2]):
        if label:
            if side == 'RIGHT':
                tiff.imsave(destination + 'a/' +
                            savename + '_' + str(s).zfill(3) + '.tif', npy[:, :, s])
            elif side == 'LEFT':
                tiff.imsave(destination + 'b/' +
                            savename + '_' + str(s).zfill(3) + '.tif', npy[:, :, s])
        else:
            if side == 'LEFT':
                tiff.imsave(destination + 'a/' +
                            savename + '_' + str(s).zfill(3) + '.tif', npy[:, :, s])
            elif side == 'RIGHT':
                tiff.imsave(destination + 'b/' +
                            savename + '_' + str(s).zfill(3) + '.tif', npy[:, :, s])
