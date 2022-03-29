import pydicom
import numpy as np
import glob
import os
import distutils
import pandas as pd

import numpy as np
from PIL import Image
import cv2
from skimage.metrics import structural_similarity as ssim
from utils.images_utils import imagesc, to_8bit


def get_tse_dess_infos(ID, meta):
    """
    read the meta data and align TSE and DESS using Slice Location
    """
    dess = meta.loc[(meta['ID'] == ID) & (meta['sequences'] == 'DESS')]
    tse = meta.loc[(meta['ID'] == ID) & (meta['sequences'] == 'TSE')]
    dess_slice_location = dess['SliceLocation'].values
    tse_slice_location = tse['SliceLocation'].values
    match_tse_dess = [np.argmin(np.abs(dess_slice_location - x)) for x in tse_slice_location]
    return tse, dess, match_tse_dess


def quick_compare(i, s, npy_folder, show=True):
    """
    show the aligned TSE and DESS side-by-side to compare
    """
    ID = ID_list[i]
    slice = s
    tse, dess, match_tse_dess = get_tse_dess_infos(ID, meta)
    t = np.load(npy_folder + tse.iloc[slice]['filename'] + '.npy')
    match_slice = match_tse_dess[slice]
    d = np.load(npy_folder + dess.iloc[match_slice]['filename'] + '.npy')
    # thresholding, normalization, cropping, and resize
    d[d >= 400] = 400
    t = t[:, 2:446]
    t = t/t.max()
    d = d/d.max()
    d = cv2.resize(d, dsize=(444, 444), interpolation=cv2.INTER_CUBIC)
    print('slice: ' + str(slice) + ', match slice: ' + str(match_slice))
    imagesc(np.concatenate([t, d], 1), show=show)
    return t, d


def make_compare(im1, im2):
    im1[im1 >= 800] = 800
    im2[im2 >= 800] = 800

    im1 = im1 - im1.min()
    if im1.max() != 0:
        im1 = im1 / im1.max()
    im2 = im2 - im2.min()
    if im2.max() != 0:
        im2 = im2 / im2.max()

    compare = np.zeros((im1.shape[0], im1.shape[1], 3))
    compare[:, :, 0] = im1
    compare[:, :, 2] = im1
    compare[:, :, 1] = im2
    return compare


def linear_registration(im1, im2, warp_mode, steps):
    # try registration using open CV
    # use cv2.findTransformECC

    #im1 = to_8bit(im1)[:, :, 0]#cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    #im2 = to_8bit(im2)[:, :, 0]#cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    if warp_mode == 3:
        warp_matrix = np.eye(3, 3, dtype=np.float32)
    else:
        warp_matrix = np.eye(2, 3, dtype=np.float32)
    number_of_iterations = steps
    termination_eps = 1e-10
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations,  termination_eps)
    (cc, warp_matrix) = cv2.findTransformECC(im1, im2, warp_matrix, warp_mode, criteria=criteria)

    sz = im1.shape
    im2_aligned = apply_warp(sz, im2, warp_matrix, warp_mode)
    return im2_aligned, warp_matrix


def apply_warp(sz, im2, warp_matrix, warp_mode):
    if warp_mode == 3:
        im2_aligned = cv2.warpPerspective(im2, warp_matrix, (sz[1], sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    else:
        im2_aligned = cv2.warpAffine(im2, warp_matrix, (sz[1], sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    return im2_aligned


a = sorted(glob.glob('/media/ghc/GHc_data1/paired_images/painpickedgood0521/train/a/*'))
b = sorted(glob.glob('/media/ghc/GHc_data1/paired_images/painpickedgood0521/train/b/*'))


if __name__ == '__main__':
    from utils.oai_unzip import dcm_to_npys_and_metas, meta_process
    ###################################################
    """
    PROJECT STRUCTURE
    data/   .dcm and extracted .npy and meta files
    outputs/ output of the registration results
    """
    ###################################################
    # CHANGE THIS TO WHERE YOU SAVE THE NUMPY FILES ###
    root_folder = '/media/ghc/GHc_data1/OAI_extracted/'
    ###################################################

    # extract dcm pixel array to the npy folder and gather metadata
    dcm_folder = root_folder + 'OAI00womac3/'
    npy_folder = root_folder + 'OAI00womac3Npy/'
    dcm_to_npys_and_metas(source=dcm_folder,
                          destination=npy_folder,
                          metas=['ImagePositionPatient', 'SliceLocation'])
    meta, ID_list = meta_process(meta=pd.read_csv(npy_folder + 'meta.csv'))

    for i in range(19):
        num_of_slice = meta.loc[(meta['ID'] == ID_list[i]) & (meta['sequences'] == 'TSE')].shape[0]
        for s in range(num_of_slice):
            warp_mode = cv2.MOTION_HOMOGRAPHY
            ###################################################
            # Add the function where you auto-match a DESS slice
            ###################################################
            t, d = quick_compare(i, s, npy_folder, show=False)
            t_aligned, ssim0, ssim1 = linear_registration(im1=d, im2=t, warp_mode=warp_mode, show=False)
            savename = ('outputs/registration/' + ID_list[i] + '_' + str(s) + '_{:.3f}_{:.3f}.jpg').format(ssim0, ssim1)
            imagesc([d, make_compare(d, t_aligned), t_aligned, t], show=False, save=savename)

