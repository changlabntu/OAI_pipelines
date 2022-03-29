import os, glob, time
import multiprocessing
import subprocess
import pydicom
import pandas as pd
import numpy as np
from PIL import Image
import shutil
#import zipfile


def zip_content_list(zipfile):
    # create the list of zip file content in a txt list
    os.system('unzip -Z1 ' + zipfile + ' "*/*/*/*" > ' + zipfile.split('.zip')[0] + '.txt')
    # read the text file to pandas
    x = pd.read_csv(zipfile.split('.zip')[0] + '.txt', header=None)

    #x = [x for x in list(x['folders']) if len(x.split('/')) == 5]
    #x = list(set([a.replace(a.split('/')[-1], '') for a in x]))
    #list_of_folders = pd.DataFrame({'folders': x})

    #
    x = sorted(list(x[0]))
    x = [y for y in x if len(y.split('/')) == 5]
    x = list(set([y[:-3] for y in x]))
    list_of_folders = pd.DataFrame({'folders': x})
    return list_of_folders


def scan_zip_locate_sequences_par(zipfile):
    """
    use scan_zip_locate_sequences in parallel
    :return:
    """
    list_of_folders = zip_content_list(zipfile)

    n_worker = 20
    range_list = np.array_split(range(list_of_folders.shape[0]), n_worker)

    workers = []
    for i in range(n_worker):
        workers.append(multiprocessing.Process(target=scan_zip_locate_sequences, args=(range_list[i], i, list_of_folders)))

    for i in range(n_worker):
        workers[i].start()

    for i in range(n_worker):
        workers[i].join()

    list_of_csv = [pd.read_csv(str(i)+'.csv') for i in range(n_worker)]
    total = pd.concat(list_of_csv)
    final = pd.DataFrame({'ID': [x.split('/')[1] for x in total['0'].values], 'folders': total['0'], 'sequences': total['1']})
    final.to_csv('meta/path' + zipfile.split('/')[-1].split('.')[0] + '.csv', index=False)
    return 0


def scan_zip_locate_sequences(irange, n_worker, list_of_folders):
    """
    open the first slice in every subfolder of the OAI zip file to find out the type of sequence (SeriesDescription)
    :param irange:
    :param n_worker:
    :return:
    """
    dir_name = 'temp' + str(n_worker) + '/'
    found = []
    for i in irange:
        tini = time.time()
        if os.path.isdir(dir_name):
            shutil.rmtree(dir_name)
        os.mkdir(dir_name)

        sub_folder = list_of_folders['folders'][i]
        for z in range(1, 10):
            try:
                subprocess.run(['unzip', '-j', zipfile, sub_folder + str(z).zfill(3), '-d', dir_name])
                found_series_description = pydicom.read_file(glob.glob(dir_name + '*')[0]).SeriesDescription
                break
            except:
                print(str(z) + '  not found')

        list_of_folders.loc[list_of_folders['folders']
                            == sub_folder, 'SeriesDescription'] = found_series_description
        found.append((sub_folder,  found_series_description))
        print(str(i) + '  ' + found_series_description + ' ' + str(time.time() - tini))

    df = pd.DataFrame(found)
    df.to_csv(str(n_worker) + '.csv')
    return found


def find_folders_by_id_and_sequence(df, path_of_sequences):
    """
    find the folder .zip file by patient ID and MRI sequence
    """
    folders = []
    for i in range(df.shape[0]):
        ID, sequences = df.iloc[i][['ID', 'sequences']]
        folders.append(path_of_sequences.loc[(path_of_sequences['ID'] == ID) & (path_of_sequences['sequences'] == sequences), 'folders'].values[0])
    return folders


def meta_process(meta):
    """
    process the meta data
    """
    meta['ID'] = [x.split('/')[1] for x in meta['filename']]
    meta['series'] = [x.split('/')[0].split('_')[3] for x in meta['filename']]
    meta['slice'] = [int(x.split('/')[2]) for x in meta['filename']]
    meta['sequences'] = [x.split('/')[0].split('_')[2] for x in meta['filename']]
    meta['side'] = [x.split('/')[0].split('_')[3] for x in meta['filename']]
    return meta


if __name__ == 'main':
    """
    locate pathes of MRI sequences and save as a table of [ID, sequences, folders] if not exist
    """
    #zipfile = '/media/ghc/GHc_data1/OAI_raw/OAI12MonthImages/results/12m.zip'
    zipfile = '/media/ghc/GHc_data2/OAI/OAI96MonthImages/results/96m.zip'
    scan_zip_locate_sequences_par(zipfile)
