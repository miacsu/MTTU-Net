import pickle
import os
import numpy as np
import nibabel as nib

grade_idh= 0
grade_1p19q= 1
grade_idh_1p19q = 2

modalities = ('flair', 't1ce', 't1', 't2')

# train
train_set = {
        'root': 'BraTS/BraTS2021/RSNA_ASNR_MICCAI_BraTS2021_TrainingData/',
        'flist': 'data_all_1.txt',
        'grade_file':'data/IDH_train_ormiss_data.csv',
        'has_label': True,
        }

# test/validation data
valid_set = {
        'root': 'BraTS/BraTS2021/RSNA_ASNR_MICCAI_BraTS2021_ValidationData/',
        'flist': 'valid_data.txt',
        'grade_file': '',#'data/IDH_test_ormiss_data.csv',
        'has_label': False
        }

test_set = {
        'root': 'path to testing set',
        'flist': 'test.txt',
        'grade_file': 'IDH_test_data.csv',
        'has_label': False
        }


def nib_load(file_name):
    if not os.path.exists(file_name):
        print('Invalid file name, can not find the file!')

    proxy = nib.load(file_name)
    data = proxy.get_data()
    proxy.uncache()
    return data

def csv_load(file_name,subject_id):
    """

    :param file_name:
    :param subject_id:
    :param return_type: grade_idh= 0  grade_1p19q= 1  grade_idh_1p19q = 2
    :return:
    """
    import pandas as pd
    if not os.path.exists(file_name):
        print('Invalid file name, can not find the file!')
    csv_data = pd.read_csv(file_name)

    grade = csv_data[csv_data.BraTS_2020_ID == subject_id]['Grade'].values.ravel()
    idh = csv_data[csv_data.BraTS_2020_ID == subject_id]['IDH_status'].values.ravel()

    if grade == 'G2' or grade == 'G3' or grade == 'LGG':
        label_grade = 0
    elif grade == 'G4' or grade == 'HGG':
        label_grade = 1
    else:
        label_grade = -1
        print("the grade of the case [{}] is {} (not in G2,G3,and G4)".format(subject_id,grade))

    if idh == 'WT':
        label_idh = 0
    elif idh == 'Mutant':
        label_idh = 1
    else:
        label_idh = -1
        print("the IDH of the case [{}] is {} (not in WT and Mutant)".format(subject_id, idh))

    return [label_grade,label_idh]

def process_i16(path, has_label=True):
    """ Save the original 3D MRI images with dtype=int16.
        Noted that no normalization is used! """
    label = np.array(nib_load(path + 'seg.nii.gz'), dtype='uint8', order='C')

    images = np.stack([
        np.array(nib_load(path + modal + '.nii.gz'), dtype='int16', order='C')
        for modal in modalities], -1)# [240,240,155]

    output = path + 'data_i16.pkl'

    with open(output, 'wb') as f:
        print(output)
        print(images.shape, type(images), label.shape, type(label))  # (240,240,155,4) , (240,240,155)
        pickle.dump((images, label), f)

    if not has_label:
        return


def process_f32b0(path, seg_label=True,grade_file=''):
    """ Save the data with dtype=float32.
        z-score is used but keep the background with zero! """
    if seg_label:
        label = np.array(nib_load(path + 'seg.nii.gz'), dtype='uint8', order='C')
    if os.path.exists(grade_file):
        subject_id = os.path.basename(os.path.dirname(path))
        subtype_label = csv_load(grade_file,subject_id)

    images = np.stack([np.array(nib_load(path + modal + '.nii.gz'), dtype='float32', order='C') for modal in modalities], -1)  # [240,240,155]

    output = path + 'data_f32b0.pkl'
    mask = images.sum(-1) > 0
    for k in range(4):

        x = images[..., k]  #
        y = x[mask]

        # 0.8885
        x[mask] -= y.mean()
        x[mask] /= y.std()

        images[..., k] = x

    with open(output, 'wb') as f:
        print(output)

        if seg_label and not os.path.exists(grade_file):
            pickle.dump((images, label), f)
        elif seg_label and os.path.exists(grade_file):
            print("{} IDH_label:{}".format(subject_id,subtype_label))
            pickle.dump((images, label, subtype_label), f)
        else:
            pickle.dump(images, f)

    if not seg_label:
        return


def doit(dset):
    root, has_label,grade_file = dset['root'], dset['has_label'],dset['grade_file']
    file_list = os.path.join(root, dset['flist'])
    subjects = open(file_list).read().splitlines()
    names = [sub.split('/')[-1] for sub in subjects]
    paths = [os.path.join(root, sub, name + '_') for sub, name in zip(subjects, names)]

    for path in paths:
        process_f32b0(path, has_label,grade_file)

if __name__ == '__main__':
    #doit(train_set)
    doit(valid_set)
    # doit(test_set)

