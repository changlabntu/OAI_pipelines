import os, glob

import numpy as np
import tifffile as tiff
import pandas as pd


def to_8bit(x):
    x = (x / x.max() * 255).astype(np.uint8)
    if len(x.shape) == 2:
        x = np.concatenate([np.expand_dims(x, 2)]*3, 2)
    return x


def make_main_csv():
    list_tif = sorted(glob.glob(root + source + '/*.tif'))
    knees = [x.split('/')[-1].split('.')[0] for x in list_tif]
    df = pd.DataFrame({'knees': knees, 'threshold': None, 'Annotated': None})
    df.to_csv(root + 'csv/temp.csv')


def create_binary_if_have_threshold(df, source, destination):
    x = df.loc[~df['threshold'].isna()]
    for i in range(x.shape[0]):
        name = x.iloc[i]['knees']
        threshold = x.iloc[i]['threshold']
        print(name)
        print(threshold)
        dcm = tiff.imread(os.path.join(root, source, name + '.tif'))
        mask = dcm * 1
        mask = (mask >= threshold) / 1
        combined = np.concatenate([dcm / dcm.max(), mask], 2)
        combined = np.concatenate([np.expand_dims(combined, 3)]*3, 3)
        combined = to_8bit(combined)
        if not os.path.isfile(os.path.join(root, destination, name + '.tif')):
            tiff.imsave(os.path.join(root, destination, name + '.tif'), combined)


root = '/home/ghc/Dropbox/Family Room/LabProjects/OAI_new_annotation/womac3_w_moaks/'
source = 'a_moaks'
df_main = pd.read_csv(root + 'csv/main.csv')

create_binary_if_have_threshold(df_main, source, destination='combined')


