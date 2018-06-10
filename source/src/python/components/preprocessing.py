_author__ = 'MSteger'

import os
import pandas as pd
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from joblib import Parallel, delayed
from components.helpers import mv_file_to_parent_dir
from glob import glob
from components.helpers import sample_from_tinyImageNet, cp_files

class tinyImageNet(Dataset):
    def __init__(self, data_dir, transform = None):
        self.filenames = [os.path.join(file[0], name) for file in os.walk(data_dir) for name in file[-1] if name.lower().endswith(('.jpeg', '.jpg', '.png'))]
        self.labels = [filename.split('/')[-2] for filename in self.filenames]
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img = Image.open(self.filenames[idx]).convert('RGB')
        if self.transform is not None: img = self.transform(img)
        return img, self.labels[idx]

class PhantDataset(tinyImageNet):
    pass

def loaders(path, dataset, transformers, **load_params):
    data_splits, data_loaders = [name for name in os.listdir(path) if os.path.isdir(os.path.join(path,name))], {}
    for split in data_splits:
        split_dir = os.path.join(path, split)
        data_loaders[split] = DataLoader(dataset = dataset(data_dir = split_dir, transform = transformers[split]), **load_params)
    return data_loaders

class tinyImageNet_Prepare(object):

    def __init__(self, path):
        self.path = path

    def df_to_dict(self, df, col = 1):
        """
        transform df to dict: index == key; col == value

        :param df:
        :param col:
        :return:
        """
        iteration = df.index.values
        return dict(zip(iteration, df.loc[iteration, col]))

    def restructure_val_folders(self):
        """
        This method is responsible for separating validation images into separate sub folders
        """
        path = os.path.join(self.path, 'val/images')  # path where validation data is present now
        filename = os.path.join(self.path, 'val/val_annotations.txt')  # file where image2class mapping is present

        file_mapping = pd.read_csv(filename, header = None, index_col = 0, usecols = [0,1], sep = '\t', lineterminator = '\n')
        file_mapping_dict = self.df_to_dict(df = file_mapping, col = 1)

        # Create folder if not present, and move image into proper folder
        for img, folder in file_mapping_dict.items():
            newpath = (os.path.join(path, folder))
            if not os.path.exists(newpath):  # check if folder exists
                os.makedirs(newpath)

            if os.path.exists(os.path.join(path, img)):  # Check if image exists in default directory
                os.rename(os.path.join(path, img), os.path.join(newpath, img))
        return

    def restructure_train_folders(self, file_endswith = '.JPEG', n_jobs = -1):
        path = os.path.join(self.path, 'train')
        filenames = [os.path.join(file[0], name) for file in os.walk(path) for name in file[-1] if name.endswith(file_endswith)]
        Parallel(n_jobs = n_jobs, backend="multiprocessing")(delayed(mv_file_to_parent_dir)(file_path = file_path) for file_path in filenames)
        return self

    def get_classes(self, classes_lst = None):
        filename = os.path.join(self.path, 'words.txt')
        if classes_lst is None: classes_lst = [cls.split('/')[-2] for cls in glob(self.path + '/train/*/')]
        class_mapping = pd.read_csv(filename, header = None, index_col = 0, sep = '\t', lineterminator = '\n')
        return self.df_to_dict(df = class_mapping.loc[classes_lst], col = 1)

def process_img(img_path, transformer, output_dir, output_filename = None):
    # TODO: add options for flips, crops & color jitter!
    try:
        img = Image.open(img_path).convert('RGB')
        new_img = transformer(img)
        new_img_filename = os.path.split(img_path)[-1]
        if output_filename is not None:
            ext = os.path.splitext(os.path.split(img_path)[-1])[-1]
            new_img_filename = '{}_{}{}'.format(output_dir.split('/')[-2], output_filename,ext.lower())
        new_img_path = os.path.join(output_dir, new_img_filename)
        new_img.save(new_img_path)
        print 'saved {} to {}!'.format(img_path, new_img_path)
    except Exception as e:
        print 'failure: {} - {}'.format(img_path, e)
    return

def run_preprocessing(img_paths, transformer, output_dir, n_jobs = 1, index_imgs = False):
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    return Parallel(n_jobs = n_jobs, backend="multiprocessing")(delayed(process_img)(img_path = img_path, transformer = transformer, output_dir = output_dir, output_filename = idx if index_imgs else None) for idx, img_path in enumerate(img_paths))

if __name__ == '__main__':
    ### checkout dataloader
    transformer = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    path = '/media/msteger/storage/resources/tiny-imagenet-200/'
    data_transformers = {'train': transformer, 'val': transformer, 'test': transformer}

    data = tinyImageNet_Prepare(path = path)
    # data.restructure_val_folders()
    # data.restructure_train_folders(n_jobs = -1)
    classes = data.get_classes(classes_lst = None)
    data_loaders = loaders(path = path, dataset = tinyImageNet, transformers = data_transformers, batch_size = 8, shuffle = True, num_workers = 4, pin_memory = True)

    ### checkout img preprocessing
    raw_crawler_images = r'/media/msteger/storage/resources/DreamPhant/downloads/'
    img_paths = [os.path.join(root, file) for root, dirnames, filenames in os.walk(raw_crawler_images) for file in filenames]
    transformer = transforms.Compose([transforms.Resize((224, 224))])
    run_preprocessing(img_paths = img_paths, transformer = transformer, output_dir = r'/media/msteger/storage/resources/DreamPhant/1/', n_jobs = -1, index_imgs = True)

    ### checkout sampling from tinyImagenet (for non elephant data)
    non_Phant_data_path = r'/media/msteger/storage/resources/tiny-imagenet-200/train'
    non_Phant_data = sample_from_tinyImageNet(data_dir = non_Phant_data_path, size = 6000, exclude_class = ['n01522450'], include_class = None, class_prob = None)
    cp_files(file_paths=non_Phant_data, new_directory_path=r'/media/msteger/storage/resources/DreamPhant/data/train/0', n_jobs=-1)

