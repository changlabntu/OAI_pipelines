import pandas as pd
from utils.oai_zip_utils import meta_process
import zipfile
from dotenv import load_dotenv
import os, glob
import pydicom
import numpy as np


def unzip_selected(df, zipname, destination):
    print('Preparing archive...')
    archive = zipfile.ZipFile(zipname)
    print('Done...')
    for i in range(df.shape[0]):
        folder = df.iloc[i]['folders']
        print(folder)
        n = 1
        while True:
            try:
                archive.extract(folder+str(n).zfill(3),  destination)
                n = n + 1
            except:
                break


def dcm_to_npys_and_metas(x, destination, metas):
    """
    extract npys from dcms and record meta
    """
    os.makedirs(destination, exist_ok=True)

    folder_list = []
    cohorts = sorted([x.split('/')[-2] for x in glob.glob(x + '*/')])
    for c in cohorts:
        folder_list = folder_list + sorted(glob.glob(x + c + '/*/*/*'))

    dcm_meta = []
    for f in folder_list[:]:
        dcm_list = glob.glob(f+'/*')
        dcm_list.sort()

        # find ID and sequence and make folders if don't exist
        ID = f.split('/')[-3]
        sequence = pydicom.read_file(dcm_list[0]).SeriesDescription
        VER = str(f.split('/')[-4].split('.')[0]).zfill(2)
        os.makedirs(destination + sequence + '/', exist_ok=True)
        os.makedirs(destination + sequence + '/' + ID + '_' + VER + '/', exist_ok=True)

        for d in dcm_list:
            dcm = pydicom.read_file(d)
            npyname = destination + sequence + '/' + ID + '_' + VER + '/' + d.split('/')[-1]
            np.save(npyname + '.npy', dcm.pixel_array)
            meta = [sequence + '/' + ID + '_' + VER + '/' + d.split('/')[-1]]
            for m in metas:
                meta = meta + [getattr(dcm, m)]
            dcm_meta.append(meta)

    dcm_meta = pd.DataFrame(dcm_meta, columns=['filename']+metas)
    dcm_meta.to_csv(destination + 'meta.csv', index=False)


def dcm_2_npys(dcm_folder):
    npy_folder = dcm_folder + 'Npy/'
    dcm_to_npys_and_metas(x=dcm_folder,
                          destination=npy_folder,
                          metas=['ImagePositionPatient', 'SliceLocation'])
    meta = meta_process(meta=pd.read_csv(npy_folder + 'meta.csv'))
    return meta


def get_zip():
    from main_oaimeta import ver_to_months
    zipfiles = dict()
    for VER in ['00', '01', '03', '05', '06', '08', '10']:
        zipfiles[VER] = os.environ.get('zip' + ver_to_months(VER) + 'm')
    return zipfiles


if __name__ == '__main__':
    # paths
    load_dotenv('.env')
    source = os.environ.get('source')
    destination = os.environ.get('destination')
    zipfiles = get_zip()

    # name of the data
    csv = pd.read_csv('meta/womac5min0.csv')
    data_name = 'womac5min0/'

    # unzip dicom files from the zip file
    for v in set(csv['VER']):
        csv_ver = csv.loc[csv['VER'] == v]
        unzip_selected(df=csv_ver.iloc[:, :],
                       zipname=source + zipfiles[str(v).zfill(2)],
                       destination=destination + data_name)

    # convert the images from the dicom to .npy
    meta = dcm_2_npys(dcm_folder=destination + data_name)