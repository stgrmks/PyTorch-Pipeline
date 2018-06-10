_author__ = 'MSteger'

import os
import glob
import numpy as np
from shutil import copyfile, move, copytree, rmtree
from itertools import compress
from joblib import Parallel, delayed

def sample_from_tinyImageNet(data_dir, size, exclude_class = None, include_class = None, class_prob = None):
    train_samples = []
    if include_class is None: include_class = [dir for dir in os.listdir(data_dir)] if class_prob is None else class_prob.keys()
    if class_prob is None: class_prob = {dir: 1./len(include_class) for dir in include_class} # equally distributed by default

    if exclude_class is not None:
        for ex_class in exclude_class: class_prob.pop(ex_class, None)

    for sample_class, sample_prob in class_prob.items():
        try:
            sample_class_path = os.path.join(data_dir, sample_class)
            class_size = int(sample_prob * size)
            all_class_samples = [f for f in glob.glob('{}/*.*'.format(sample_class_path)) if f.lower().endswith(('.jpeg', '.jpg', '.jpeg', '.png'))]
            random_selection = np.random.randint(0, len(all_class_samples), class_size)
            train_samples += list(compress(all_class_samples, random_selection))
        except Exception as e:
            print 'failure! {}'.format(e)
    return train_samples

def cp_file(old_path, new_path):
    print 'cp {} to {}'.format(old_path, new_path)
    return copyfile(old_path, new_path)

def mv_file(old_path, new_path):
    print 'mv {} to {}'.format(old_path, new_path)
    try:
        move(old_path, new_path)
    except Exception as e:
        print 'failure! {}'.format(e)
    return

def cp_files(file_paths, new_directory_path, n_jobs = 1, fn = cp_file):
    if not os.path.exists(new_directory_path): os.makedirs(new_directory_path)
    return Parallel(n_jobs = n_jobs, backend = 'multiprocessing')(delayed(fn)(old_path = old_path, new_path = os.path.join(new_directory_path, '0_{}{}'.format(idx, os.path.splitext(old_path)[-1]))) for idx, old_path in enumerate(file_paths))

def mv_file_to_parent_dir(file_path, rm_folder_in_path = 'images'):
    f = file_path.split('/')
    if rm_folder_in_path in f:
        f.remove(rm_folder_in_path)
    else:
        return
    new_path = os.path.join(*['/'] + f)
    if not os.path.isfile(new_path): os.rename(file_path, new_path)
    return

def train_test_split(train_dir, test_dir, test_size = 0.1, stratify = False, n_jobs = 1):
    if not os.path.exists(test_dir): os.makedirs(test_dir)
    train_set_meta = {}
    for dir in os.listdir(train_dir):
        cwd = os.path.join(train_dir, dir)
        samples_in_class = [os.path.join(cwd, name) for name in os.listdir(cwd) if os.path.isfile(os.path.join(cwd, name))]
        train_set_meta[dir] = {'len': len(samples_in_class), 'paths': samples_in_class}

    if stratify:
        sample_sizes = dict(zip(train_set_meta.keys(), np.dot([train_set_meta.get(k, {}).get('len') for k in train_set_meta.keys()], test_size).astype(int)))
    else:
        sample_sizes = dict(zip(train_set_meta.keys(), [int(np.sum([train_set_meta.get(k, {}).get('len') for k in train_set_meta.keys()]) * test_size / len(train_set_meta))] * len(train_set_meta)))

    for c_name, c_samples in train_set_meta.items():
        cwd = os.path.join(test_dir, c_name)
        if not os.path.exists(cwd): os.makedirs(cwd)
        random_selection = np.random.randint(0, c_samples['len'], sample_sizes[c_name])
        sel_samples = list(compress(c_samples['paths'], random_selection))
        Parallel(n_jobs = n_jobs, backend='multiprocessing')(delayed(mv_file)(old_path = old_path, new_path = old_path.replace(train_dir, test_dir)) for old_path in sel_samples)

    return

def assemble_datasets(data_path, datasets_folder, **split_params):
    if os.path.exists(datasets_folder): rmtree(datasets_folder)
    os.makedirs(datasets_folder)
    copytree(data_path, os.path.join(datasets_folder, 'train'))
    return train_test_split(train_dir = os.path.join(datasets_folder, 'train'), test_dir = os.path.join(*['/']+os.path.join(datasets_folder, 'train').split('/')[1:-1] + ['val']), **split_params)

def geo_mean(iterable):
    a = np.array(iterable)
    return a.prod()**(1.0/len(a))

if __name__ == '__main__':
    assemble_datasets(data_path = r'/media/msteger/storage/resources/DreamPhant/data_all', datasets_folder = '/media/msteger/storage/resources/DreamPhant/datasets', stratify = False, test_size = 0.3, n_jobs = -1)
    print 'done'