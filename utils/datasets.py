import logging
import os
from abc import ABC
from typing import Tuple, Any
import random
import numpy as np
import torch
import torchvision
from pandas import read_csv
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CIFAR10, CIFAR100
from torchvision.datasets.folder import pil_loader, accimage_loader
from torchvision.transforms import transforms
from tqdm import tqdm

import configs
from functions.evaluate_roxf import configdataset, DATASETS
from functions.mining import SimpleMemoryBank
from utils.augmentations import GaussianBlurOpenCV
import scipy.io as sio

DATA_FOLDER = {
    'nuswide': 'data/nuswide_v2_256_resize',  # resize to 256x256
    'imagenet': 'data/imagenet_resize',  # resize to 224x224
    'cifar': 'data/cifar',  # auto generate
    'coco': 'data/coco',
    'gldv2': 'data/gldv2delgembed',
    'roxf': 'data/roxford5kdelgembed',
    'rpar': 'data/rparis5kdelgembed',
    'awa2': '../data/AwA2/Animals_with_Attributes2',
    'cub': '../data/CUB_200_2011',
    'sun': '../data/SUN/'
}

class BaseDataset(Dataset, ABC):
    def get_img_paths(self):
        raise NotImplementedError


class HashingDataset(BaseDataset):
    def __init__(self, root,
                 transform=None,
                 target_transform=None,
                 filename='train',
                 separate_multiclass=False,
                 ratio=1):
        if torchvision.get_image_backend() == 'PIL':
            self.loader = pil_loader
        else:
            self.loader = accimage_loader

        self.separate_multiclass = separate_multiclass
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.filename = filename
        self.train_data = []
        self.train_labels = []
        self.ratio = ratio

        filename = os.path.join(self.root, self.filename)

        is_pkl = False

        with open(filename, 'r') as f:
            while True:
                lines = f.readline()
                if not lines:
                    break

                path_tmp = lines.split()[0]
                label_tmp = lines.split()[1:]
                self.is_onehot = len(label_tmp) != 1
                if not self.is_onehot:
                    label_tmp = lines.split()[1]
                if self.separate_multiclass:
                    assert self.is_onehot, 'if multiclass, please use onehot'
                    nonzero_index = np.nonzero(np.array(label_tmp, dtype=np.int))[0]
                    for c in nonzero_index:
                        self.train_data.append(path_tmp)
                        label_tmp = ['1' if i == c else '0' for i in range(len(label_tmp))]
                        self.train_labels.append(label_tmp)
                else:
                    self.train_data.append(path_tmp)
                    self.train_labels.append(label_tmp)

                is_pkl = path_tmp.endswith('.pkl')  # if save as pkl, pls make sure dont use different style of loading

        if is_pkl:
            self.loader = torch.load

        self.train_data = np.array(self.train_data)
        self.train_labels = np.array(self.train_labels, dtype=float)

        if ratio != 1:
            assert 0 < ratio < 1, 'data ratio is in between 0 and 1 exclusively'
            N = len(self.train_data)
            randidx = np.arange(N)
            np.random.shuffle(randidx)
            randidx = randidx[:int(ratio * N)]
            self.train_data = self.train_data[randidx]
            self.train_labels = self.train_labels[randidx]

        logging.info(f'Number of data: {self.train_data.shape[0]}')

    def filter_classes(self, classes):  # only work for single class dataset
        new_data = []
        new_labels = []

        for idx, c in enumerate(classes):
            new_onehot = np.zeros(len(classes))
            new_onehot[idx] = 1
            cmask = self.train_labels.argmax(axis=1) == c

            new_data.append(self.train_data[cmask])
            new_labels.append(np.repeat([new_onehot], int(np.sum(cmask)), axis=0))
            # new_labels.append(self.train_labels[cmask])

        self.train_data = np.concatenate(new_data)
        self.train_labels = np.concatenate(new_labels)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.train_data[index], self.train_labels[index]
        target = torch.tensor(target)

        img = self.loader(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

    def __len__(self):
        return len(self.train_data)

    def get_img_paths(self):
        return self.train_data


class IndexDatasetWrapper(BaseDataset):
    def __init__(self, ds) -> None:
        super(Dataset, self).__init__()
        self.__dict__['ds'] = ds

    def __setattr__(self, name, value):
        setattr(self.ds, name, value)

    def __getattr__(self, attr):
        if attr in self.__dict__:
            return getattr(self, attr)
        return getattr(self.ds, attr)

    def __getitem__(self, index: int) -> Tuple:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target, index) where target is index of the target class.
        """
        outs = self.ds.__getitem__(index)
        return tuple(list(outs) + [index])

    def __len__(self):
        return len(self.ds)

    def get_img_paths(self):
        return self.ds.get_img_paths()


class Denormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


class AwA2Dataset(Dataset):

    feature_img_names = []
    with open("../data/AwA2/features_alexnet_imgnames.txt",'r') as f:
        for line in f.readlines():
            feature_img_names.append(line.strip().replace("AWA2","AwA2").replace("2//J","2/J"))
    features_alexnet = np.load("../data/AwA2/features_alexnet_AWA2.npy")

    def __init__(self, root,transform=None,filename='train'):
        self.transform = transform
        self.filename = filename
        self.loader = pil_loader
        self.root = os.path.expanduser(root)

        #unseen 20%
        self.seen_class_path = os.path.join(self.root, 'trainclasses.txt')
        self.unseen_class_path = os.path.join(self.root, 'testclasses.txt')
        # #unseen 40%
        # self.seen_class_path = os.path.join(self.root, 'trainclasses40.txt')
        # self.unseen_class_path = os.path.join(self.root, 'testclasses40.txt')
        # #unseen 60%
        # self.seen_class_path = os.path.join(self.root, 'trainclasses60.txt')
        # self.unseen_class_path = os.path.join(self.root, 'testclasses60.txt')


        self.img_dir = os.path.join(self.root,"JPEGImages")
        self.class_id_path = os.path.join(self.root,"classes.txt")
        self.attr_path = os.path.join(self.root,"predicate-matrix-continuous.txt")

        self.all_class_names = []
        self.use_classes = []

        self.init_data()

    def init_data(self):
        with open(self.class_id_path, 'r') as f:
            lines = f.readlines()
        for line in lines:
            idx,name = line.strip().split('\t')
            #print(idx,name) #代码从0开始因此实际id会-1
            self.all_class_names.append(name)
        print("Load classes: ", len(self.all_class_names))

        seen_classes = []
        unseen_classes = []
        if 'train' in self.filename:
            with open(self.seen_class_path, 'r') as f:
                lines = f.readlines()
            seen_classes = [line.strip() for line in lines]
        elif 'test' in self.filename:
            with open(self.unseen_class_path, 'r') as f:
                lines = f.readlines()
            unseen_classes = [line.strip() for line in lines]
        elif 'database' in self.filename:
            with open(self.seen_class_path, 'r') as f:
                lines = f.readlines()
            seen_classes = [line.strip() for line in lines]
            with open(self.unseen_class_path, 'r') as f:
                lines = f.readlines()
            unseen_classes = [line.strip() for line in lines]
        else:
            raise Excepetion("Unkown filename")

        total_classes = seen_classes+unseen_classes
        print(self.filename, " count classes: ", len(total_classes))
        total_path = []
        total_labels = []

        self.train_data = []
        self.train_labels = []
        for classname in total_classes:
            class_dir = os.path.join(self.root,'JPEGImages',classname)
            img_names = os.listdir(class_dir)
            random.shuffle(img_names)
            if 'database' in self.filename:
                use_names = img_names[100:]
            else:
                use_names = img_names[:100]

            for name in use_names:
                img_path = os.path.join(class_dir,name)
                img_label = [0 for _ in range(len(self.all_class_names))]
                img_label[self.all_class_names.index(classname)] = 1
                self.train_data.append(img_path)
                self.train_labels.append(img_label)
        
        self.train_data = np.array(self.train_data)
        self.train_labels = np.array(self.train_labels, dtype=np.float)
        print(f'Number of data: {self.train_data.shape[0]}')


        ###load attr
        attr_data = []
        #print(self.attr_path)
        with open(self.attr_path,'r') as f:
            lines = f.readlines()
            #print(len(lines))
            for line in lines:
                values = [float(x) for x in line.strip().replace("    "," ").replace("   "," ").replace("  "," ").split(" ")]
                #参考2020nipt读取的mat，反推出的归一化方式
                mean = np.mean(values)
                std_dev = np.std(values)
                standardized_array = (values - mean) / std_dev
                attr_data.append(standardized_array)
        attr_data = np.array(attr_data) #50x85
        # print(attr_data)
        # bb
        #这里顺序应该和classes一致
        #print(self.all_class_names, seen_classes)
        with open(self.seen_class_path, 'r') as f:
            lines = f.readlines()
        seen_classes = [line.strip() for line in lines]
        with open(self.unseen_class_path, 'r') as f:
            lines = f.readlines()
        unseen_classes = [line.strip() for line in lines]
        total_classes = seen_classes+unseen_classes
        seen_id = [self.all_class_names.index(name) for name in total_classes]
        self.attr_data = attr_data[seen_id] #40x85for train 改成直接取所有类50x85
        # print(self.attr_data.shape)
        # bb

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        # print(self.train_data[index])
        # print(AwA2Dataset.features_alexnet[AwA2Dataset.feature_img_names.index(self.train_data[index])])
        # bbb
        img, target = self.train_data[index], self.train_labels[index]
        target = torch.tensor(target)

        img = self.loader(img)

        #print("1111",img.size)
        if self.transform is not None:
            img = self.transform(img)
        #print("2222",img.shape)
        shallow_feature = AwA2Dataset.features_alexnet[AwA2Dataset.feature_img_names.index(self.train_data[index])]

        return img, target, index, self.train_data[index], shallow_feature

    def __len__(self):
        return len(self.train_data)

class CubDataset(Dataset):

    feature_img_names = []
    with open("../data/CUB/features_alexnet_imgnames.txt",'r') as f:
        for line in f.readlines():
            feature_img_names.append(line.strip())
    features_alexnet = np.load("../data/CUB/features_alexnet_CUB.npy")


    def __init__(self, root,
                 transform=None,
                 filename='train'):
        self.transform = transform
        self.filename = filename
        self.loader = pil_loader
        self.root = os.path.expanduser(root)
        self.seen_unseen_path = os.path.join(self.root, 'train_test_split.txt')
        self.img_dir = os.path.join(self.root,"images")
        self.class_id_path = os.path.join(self.root,"classes.txt")
        self.attr_path = os.path.join(self.root,"attributes","class_attribute_labels_continuous.txt")

        self.all_class_names = []
        self.use_classes = []

        self.init_data()

    def init_data(self):
        with open(self.class_id_path, 'r') as f:
            class_lines = f.readlines()
        for line in class_lines:
            idx,name = line.strip().split(' ')
            #print(idx,name) #代码从0开始因此实际id会-1
            self.all_class_names.append(name)
        print("Load classes: ", len(self.all_class_names))

        split_num = 150 #150     160-120-80
        seen_classes = []
        unseen_classes = []
        if 'train' in self.filename:
            seen_classes = [line.strip().split(' ')[1] for line in class_lines[:split_num]]
        elif 'test' in self.filename:
            unseen_classes = [line.strip().split(' ')[1] for line in class_lines[split_num:]]
        elif 'database' in self.filename:
            seen_classes = [line.strip().split(' ')[1] for line in class_lines[:split_num]]
            unseen_classes = [line.strip().split(' ')[1] for line in class_lines[split_num:]]
        else:
            raise Excepetion("Unkown filename")

        total_classes = seen_classes+unseen_classes
        print(self.filename, " count classes: ", len(total_classes))
        total_path = []
        total_labels = []

        self.train_data = []
        self.train_labels = []
        for classname in total_classes:
            class_dir = os.path.join(self.root,'images',classname)
            img_names = os.listdir(class_dir)
            random.shuffle(img_names)
            if 'database' in self.filename:
                use_names = img_names[30:]
            else:
                use_names = img_names[:30]

            for name in use_names:
                img_path = os.path.join(class_dir,name)
                img_label = [0 for _ in range(len(self.all_class_names))]
                img_label[self.all_class_names.index(classname)] = 1
                self.train_data.append(img_path)
                self.train_labels.append(img_label)
        
        self.train_data = np.array(self.train_data)
        self.train_labels = np.array(self.train_labels, dtype=np.float)
        print(f'Number of data: {self.train_data.shape[0]}')


        ###load attr
        attr_data = []
        #print(self.attr_path)
        with open(self.attr_path,'r') as f:
            lines = f.readlines()
            #print(len(lines))
            for line in lines:
                values = [float(x) for x in line.strip().replace("    "," ").replace("   "," ").replace("  "," ").split(" ")]
                #参考2020nipt读取的mat，反推出的归一化方式
                mean = np.mean(values)
                std_dev = np.std(values)
                standardized_array = (values - mean) / std_dev
                attr_data.append(standardized_array)
        attr_data = np.array(attr_data) #50x85
        # print(attr_data)
        # bb
        #这里顺序应该和classes一致
        #print(self.all_class_names, seen_classes)
        
        
        seen_classes = [line.strip().split(' ')[1] for line in class_lines[:split_num]]
        unseen_classes = [line.strip().split(' ')[1] for line in class_lines[split_num:]]
        # print("--------",len(seen_classes), len(unseen_classes))
        # bb
        total_classes = seen_classes+unseen_classes
        seen_id = [self.all_class_names.index(name) for name in total_classes]
        self.attr_data = attr_data[seen_id] #40x85for train 改成直接取所有类50x85
        # print(self.attr_data.shape)
        # bb

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.train_data[index], self.train_labels[index]
        target = torch.tensor(target)

        img = self.loader(img)

        #print("1111",img.size)
        if self.transform is not None:
            img = self.transform(img)
        #print("2222",img.shape)
        shallow_feature = CubDataset.features_alexnet[CubDataset.feature_img_names.index(self.train_data[index])]
        return img, target, index, self.train_data[index], shallow_feature

    def __len__(self):
        return len(self.train_data)


class SunDataset(Dataset):
    feature_img_names = []
    with open("../data/SUN/features_alexnet_imgnames.txt",'r') as f:
        for line in f.readlines():
            feature_img_names.append(line.strip())
    features_alexnet = np.load("../data/SUN/features_alexnet_SUN.npy")
    
    def __init__(self, root,
                 transform=None,
                 filename='train'):
        self.transform = transform
        self.filename = filename
        self.loader = pil_loader
        self.root = os.path.expanduser(root)
        self.img_paths = os.path.join(self.root,'SUNAttributeDB', 'images.mat')
        self.img_dir = os.path.join(self.root,"images")
        self.attr_path = os.path.join(self.root,"SUNAttributeDB","attributeLabels_continuous.mat")

        self.all_class_names = []

        self.init_data()

    def init_data(self):
        img_paths = sio.loadmat(self.img_paths)['images']
        img_paths = [x[0][0] for x in img_paths]
        basenames = [os.path.basename(x) for x in img_paths]
        # print(len(img_paths),img_paths[:3])
        #print(img_paths[:-5000])
        #14340 ['a/abbey/sun_aakbdcgfpksytcwj.jpg', 'a/abbey/sun_aaoktempcmudsvna.jpg', 'a/abbey/sun_abegcweqnetpdlrh.jpg']
        #self.all_class_names = list(set([os.path.dirname(img_path) for img_path in img_paths]))
        #print(len(self.all_class_names),self.all_class_names[:3])
        self.all_class_names = []
        self.all_dirnames = []
        for img_path in img_paths:
            dir_names = img_path.split("/")
            #print(img_path,dir_names)
            #a/abbey/sun_aakbdcgfpksytcwj.jpg ['a', 'abbey', 'sun_aakbdcgfpksytcwj.jpg']
            if len(dir_names)==3:
                self.all_class_names.append(dir_names[1])
            elif len(dir_names)==4:
                self.all_class_names.append("_".join(dir_names[1:3]))
                # print("_".join(dir_names[1:3]))
                # bb
            else:
                raise Excepetion("unknown dir names len")
            self.all_dirnames.append(os.path.dirname(img_path))

        self.all_class_names_set = []
        #set无序无法复现
        for name in self.all_class_names:
            if name in self.all_class_names_set:
                continue
            else:
                self.all_class_names_set.append(name)
        self.all_class_names = self.all_class_names_set

        self.all_dirnames_set = []
        for name in self.all_dirnames:
            if name in self.all_dirnames_set:
                continue
            else:
                self.all_dirnames_set.append(name)
        self.all_dirnames = self.all_dirnames_set

        print("Load classes: ", len(self.all_class_names))
        # print(self.all_class_names) 
        #'abbey', 'access_road', 'airfield', 'airlock', 'airplane_cabin', 'airport_airport',
        # print(self.all_dirnames)
        #['a/abbey', 'a/access_road', 'a/airfield', 'a/airlock', 'a/airplane_cabin', 'a/airport/airport'
        split_num = 574#500    574-430-287
        seen_classes = []
        unseen_classes = []
        if 'train' in self.filename:
            seen_classes = self.all_class_names[:split_num]
        elif 'test' in self.filename:
            unseen_classes = self.all_class_names[split_num:]
        elif 'database' in self.filename:
            seen_classes = self.all_class_names[:split_num]
            unseen_classes = self.all_class_names[split_num:]
        else:
            raise Excepetion("Unkown filename")

        total_classes = seen_classes+unseen_classes
        print(self.filename, " count classes: ", len(total_classes))
        total_path = []
        total_labels = []

        self.train_data = []
        self.train_labels = []

        for classname in total_classes:
            class_id = self.all_class_names.index(classname)
            sub_dir = self.all_dirnames[class_id]
            class_dir = os.path.join(self.img_dir,sub_dir)
            img_names = os.listdir(class_dir)
            assert 20<=len(img_names)<25
            img_names = [name for name in img_names if (name in basenames)]
            # print(class_dir,len(img_names))
            # bb
            assert 20==len(img_names)
            
            # img_names = [img_path for img_path in img_paths if classname in img_path]
            random.shuffle(img_names)
            if 'database' in self.filename:
                use_names = img_names[10:]
            else:
                use_names = img_names[:10]

            for name in use_names:
                img_path = os.path.join(class_dir,name)
                img_label = [0 for _ in range(len(self.all_class_names))]
                img_label[class_id] = 1
                self.train_data.append(img_path)
                self.train_labels.append(img_label)
        
        self.train_data = np.array(self.train_data)
        self.train_labels = np.array(self.train_labels, dtype=np.float)
        print(f'Number of data: {self.train_data.shape[0]}')


        ###load attr
        img_attr_data = sio.loadmat(self.attr_path)['labels_cv']
        self.attr_data = []
        class_attr = [img_attr_data[0]]
        #print(class_attr)
        for i in range(1,len(img_paths)):
            if os.path.dirname(img_paths[i]) != os.path.dirname(img_paths[i-1]):
                #print(class_attr)
                class_attr_mean = np.mean(class_attr,axis=0)
                self.attr_data.append(class_attr_mean)
            class_attr.append(img_attr_data[i])
        class_attr_mean = np.mean(class_attr,axis=0)
        self.attr_data.append(class_attr_mean)
        self.attr_data = np.array(self.attr_data)
        # print(self.attr_data.shape)  #(717, 102)
        # bb
        #这里顺序应该和classes一致
        #print(self.all_class_names, seen_classes)
        # seen_classes = self.all_class_names[:500]
        # unseen_classes = self.all_class_names[500:]
        # total_classes = seen_classes+unseen_classes
        # seen_id = [self.all_class_names.index(name) for name in total_classes]
        # self.attr_data = attr_data[seen_id] #40x85for train 改成直接取所有类50x85
        # print(self.attr_data.shape)
        # bb

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.train_data[index], self.train_labels[index]

        target = torch.tensor(target)

        img = self.loader(img)

        #print("1111",img.size)
        if self.transform is not None:
            img = self.transform(img)
        #print("2222",img.shape)
        shallow_feature = SunDataset.features_alexnet[SunDataset.feature_img_names.index(self.train_data[index])]
        return img, target, index, self.train_data[index], shallow_feature

    def __len__(self):
        return len(self.train_data)

class InstanceDiscriminationDataset(BaseDataset):
    def augment_image(self, img):
        # if use this, please run script with --no-aug and --gpu-mean-transform
        return self.transform(self.to_pil(img))

    def weak_augment_image(self, img):
        # if use this, please run script with --no-aug and --gpu-mean-transform
        return self.weak_transform(self.to_pil(img))

    def __init__(self, ds, tmode='simclr', imgsize=224, weak_mode=0) -> None:
        super(Dataset, self).__init__()
        self.__dict__['ds'] = ds

        if 'simclr' in tmode:
            s = 0.5
            size = imgsize
            color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
            data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=size, scale=(0.5, 1.0)),
                                                  transforms.RandomHorizontalFlip(),
                                                  transforms.RandomApply([color_jitter], p=0.7),
                                                  transforms.RandomGrayscale(p=0.2),
                                                  GaussianBlurOpenCV(kernel_size=3),
                                                  # GaussianBlur(kernel_size=int(0.1 * size)),
                                                  transforms.ToTensor(),
                                                  # 0.2 * 224 = 44 pixels
                                                  transforms.RandomErasing(p=0.2, scale=(0.02, 0.2))])
            self.transform = data_transforms

        # lazy fix, can be more pretty and general, cibhash part 1/2
        elif tmode == 'cibhash':
            logging.info('CIBHash Augmentations')
            s = 0.5
            size = imgsize
            color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
            data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=size, scale=(0.5, 1.0)),
                                                  transforms.RandomHorizontalFlip(),
                                                  transforms.RandomApply([color_jitter], p=0.7),
                                                  transforms.RandomGrayscale(p=0.2),
                                                  GaussianBlurOpenCV(kernel_size=3),
                                                  # GaussianBlur(kernel_size=3),
                                                  transforms.ToTensor()])
            self.transform = data_transforms

        else:
            raise ValueError(f'unknown mode {tmode}')

        if weak_mode == 1:
            logging.info(f'Weak mode {weak_mode} activated.')
            self.weak_transform = transforms.Compose([
                transforms.Resize(256),  # temp lazy hard code
                transforms.CenterCrop(imgsize),
                transforms.ToTensor()
            ])
        elif weak_mode == 2:
            logging.info(f'Weak mode {weak_mode} activated.')
            self.weak_transform = transforms.Compose([
                transforms.Resize(256),  # temp lazy hard code
                transforms.RandomCrop(imgsize),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()
            ])

        self.weak_mode = weak_mode
        self.tmode = tmode
        self.imgsize = imgsize
        self.to_pil = transforms.ToPILImage()

    def __setattr__(self, name, value):
        setattr(self.ds, name, value)

    def __getattr__(self, attr):
        if attr in self.__dict__:
            return getattr(self, attr)
        return getattr(self.ds, attr)

    def __getitem__(self, index: int) -> Tuple[Any, Any, Any]:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target, index) where target is index of the target class.
        """
        out = self.ds.__getitem__(index)
        img, target = out[:2]  # exclude index

        # if self.tmode == 'simclr':
        #     aug_imgs = [img, self.augment_image(img)]
        # else:
        if self.weak_mode != 0:
            aug_imgs = [self.weak_augment_image(img), self.augment_image(img)]
        else:
            aug_imgs = [self.augment_image(img), self.augment_image(img)]

        return torch.stack(aug_imgs, dim=0), target, index

    def __len__(self):
        return len(self.ds)

    def get_img_paths(self):
        return self.ds.get_img_paths()


class RotationDataset(BaseDataset):

    @staticmethod
    def rotate_img(img, rot):
        img = np.transpose(img.numpy(), (1, 2, 0))
        if rot == 0:  # 0 degrees rotation
            out = img
        elif rot == 90:  # 90 degrees rotation
            out = np.flipud(np.transpose(img, (1, 0, 2)))
        elif rot == 180:  # 90 degrees rotation
            out = np.fliplr(np.flipud(img))
        elif rot == 270:  # 270 degrees rotation / or -90
            out = np.transpose(np.flipud(img), (1, 0, 2))
        else:
            raise ValueError('rotation should be 0, 90, 180, or 270 degrees')
        return torch.from_numpy(np.transpose(out, (2, 0, 1)).copy())

    def __init__(self, ds) -> None:
        super(Dataset, self).__init__()
        self.__dict__['ds'] = ds

    def __setattr__(self, name, value):
        setattr(self.ds, name, value)

    def __getattr__(self, attr):
        if attr in self.__dict__:
            return getattr(self, attr)
        return getattr(self.ds, attr)

    def __getitem__(self, index: int) -> Tuple[Any, Any, Any, Any]:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target, index) where target is index of the target class.
        """
        out = self.ds.__getitem__(index)
        img, target = out[:2]  # exclude index

        # rot_label = np.random.randint(0, 4)  # .item()
        rot_labels = [0, 1, 2, 3]

        rots = [0, 90, 180, 270]
        # rots = [0, rots[rot_label]]
        rot_imgs = [self.rotate_img(img, rot) for rot in rots]

        return torch.stack(rot_imgs, dim=0), torch.tensor(rot_labels), target, index

    def __len__(self):
        return len(self.ds)

    def get_img_paths(self):
        return self.ds.get_img_paths()


class LandmarkDataset(BaseDataset):
    def __init__(self, root,
                 transform=None,
                 target_transform=None,
                 filename='train.csv',
                 onehot=False, return_id=False):
        self.loader = pil_loader
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.filename = filename
        self.train_labels = []
        self.set_name = filename[:-4]
        self.onehot = onehot
        self.return_id = return_id

        def get_path(i: str):
            return os.path.join(root, self.set_name, i[0], i[1], i[2], i + ".jpg")

        filename = os.path.join(self.root, self.filename)
        self.df = read_csv(filename)
        self.df['path'] = self.df['id'].apply(get_path)
        self.max_index = self.df['landmark_id'].max() + 1

        logging.info(f'Number of data: {len(self.df)}')

    def to_onehot(self, i):
        t = torch.zeros(self.max_index)
        t[i] = 1
        return t

    def __getitem__(self, index):
        img = self.df['path'][index]

        if self.onehot:
            target = self.to_onehot(self.df['landmark_id'][index])
        else:
            target = self.df['landmark_id'][index]
        # target = torch.tensor(target)

        img = self.loader(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.return_id:
            return img, target, (self.df['id'][index], index)
        return img, target

    def __len__(self):
        return len(self.df)

    def get_img_paths(self):
        return self.df['path'].to_numpy()


class SingleIDDataset(BaseDataset):
    """Dataset with only single class ID
    To be merge with Landmark"""

    def __init__(self, root,
                 transform=None,
                 target_transform=None,
                 filename='train.csv',
                 onehot=False):
        self.loader = pil_loader
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.filename = filename
        self.train_labels = []
        self.set_name = filename[:-4]
        self.onehot = onehot

        def get_path(i: str):
            return os.path.join(root, "imgs", i)

        filename = os.path.join(self.root, self.filename)
        self.df = read_csv(filename)
        self.df['path'] = self.df['path'].apply(get_path)
        self.max_index = self.df['class_id'].max() + 1

        logging.info(f'Number of data: {len(self.df)}')

    def to_onehot(self, i):
        t = torch.zeros(self.max_index)
        t[i] = 1
        return t

    def __getitem__(self, index):
        img = self.df['path'][index]

        if self.onehot:
            target = self.to_onehot(self.df['class_id'][index])
        else:
            target = self.df['class_id'][index]
        # target = torch.tensor(target)

        img = self.loader(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target, index

    def __len__(self):
        return len(self.df)

    def get_img_paths(self):
        return self.df['path'].to_numpy()


class ROxfordParisDataset(BaseDataset):
    def __init__(self,
                 dataset='roxford5k',
                 filename='test.txt',
                 transform=None,
                 target_transform=None):
        self.loader = pil_loader
        self.transform = transform
        self.target_transform = target_transform
        assert filename in ['test.txt', 'database.txt']
        self.set_name = filename
        assert dataset in DATASETS
        self.cfg = configdataset(dataset, os.path.join('data'))

        logging.info(f'Number of data: {self.__len__()}')

    def __getitem__(self, index):
        if self.set_name == 'database.txt':
            img = self.cfg['im_fname'](self.cfg, index)
        elif self.set_name == 'test.txt':
            img = self.cfg['qim_fname'](self.cfg, index)

        img = self.loader(img)
        if self.set_name == 'test.txt':
            img = img.crop(self.cfg['gnd'][index]['bbx'])

        if self.transform is not None:
            img = self.transform(img)

        return img, index, index  # img, None, index is throw error

    def __len__(self):
        if self.set_name == 'test.txt':
            return self.cfg['nq']
        elif self.set_name == 'database.txt':
            return self.cfg['n']

    def get_img_paths(self):
        raise NotImplementedError('Not supported.')


class DescriptorDataset(BaseDataset):
    def __init__(self, root, filename, ratio=1):
        self.data_dict = torch.load(os.path.join(root, filename), map_location=torch.device('cpu'))
        self.filename = filename
        self.root = root
        self.ratio = ratio

        if ratio != 1:
            assert 0 < ratio < 1, 'data ratio is in between 0 and 1 exclusively'
            N = len(self.data_dict['codes'])
            randidx = np.arange(N)
            np.random.shuffle(randidx)
            randidx = randidx[:int(ratio * N)]
            for key in self.data_dict:
                self.data_dict[key] = self.data_dict[key][randidx]

        logging.info(f'Number of data in {filename}: {self.__len__()}')

    def __getitem__(self, index):
        embed = self.data_dict['codes'][index]
        label = self.data_dict['labels'][index]  # label is 1 indexed, convert to 0-indexed

        return embed, label, index  # img, None, index is throw error

    def __len__(self):
        return len(self.data_dict['codes'])

    def get_img_paths(self):
        raise NotImplementedError('Not supported for descriptor dataset. Please try usual Image Dataset if you want to get all image paths.')


class EmbeddingDataset(BaseDataset):
    def __init__(self, root,
                 filename='train.txt'):
        self.data_dict = torch.load(os.path.join(root, filename), map_location=torch.device('cpu'))
        self.filename = filename
        self.root = root
        logging.info(f'Number of data in {filename}: {self.__len__()}')

    def __getitem__(self, index):
        embed = self.data_dict['codes'][index]
        if self.filename == 'train.txt':
            label = self.data_dict['labels'][index] - 1  # label is 1 indexed, convert to 0-indexed
        else:
            label = 0
        landmark_id = self.data_dict['id'][index]

        return embed, label, (landmark_id, index)  # img, None, index is throw error

    def __len__(self):
        return len(self.data_dict['id'])

    def get_img_paths(self):
        raise NotImplementedError('Not supported for descriptor dataset. Please try usual Image Dataset if you want to get all image paths.')


class NeighbourDatasetWrapper(BaseDataset):
    def __init__(self, ds, model, config) -> None:
        super(Dataset, self).__init__()
        self.ds = ds

        device = config['device']
        loader = DataLoader(ds, config['batch_size'],
                            shuffle=False,
                            drop_last=False,
                            num_workers=os.cpu_count())

        model.eval()
        pbar = tqdm(loader, desc='Obtain Codes', ascii=True, bar_format='{l_bar}{bar:10}{r_bar}',
                    disable=configs.disable_tqdm)
        ret_feats = []

        for i, (data, labels, index) in enumerate(pbar):
            with torch.no_grad():
                data, labels = data.to(device), labels.to(device)
                x, code_logits, b = model(data)[:3]
                ret_feats.append(x.cpu())

        ret_feats = torch.cat(ret_feats)

        mbank = SimpleMemoryBank(len(self.ds), model.backbone.in_features, device)
        mbank.update(ret_feats)

        neighbour_topk = config['dataset_kwargs'].get('neighbour_topk', 5)
        indices = mbank.mine_nearest_neighbors(neighbour_topk)

        self.indices = indices[:, 1:]  # exclude itself

    def __getitem__(self, index: int) -> Tuple[Any, Any, Any, Any, Any, Any]:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target, index) where target is index of the target class.
        """
        img, target = self.ds.__getitem__(index)

        randidx = np.random.choice(self.indices[index], 1)[0]
        nbimg, nbtar = self.ds.__getitem__(randidx)

        return img, target, index, nbimg, nbtar, randidx

    def __len__(self):
        return len(self.ds)

    def get_img_paths(self):
        return self.ds.get_img_paths()


def one_hot(nclass):
    def f(index):
        index = torch.tensor(int(index)).long()
        return torch.nn.functional.one_hot(index, nclass)

    return f


# def cifar(nclass, **kwargs):
#     transform = kwargs['transform']
#     ep = kwargs['evaluation_protocol']
#     fn = kwargs['filename']
#     reset = kwargs['reset']

#     CIFAR = CIFAR10 if int(nclass) == 10 else CIFAR100
#     traind = CIFAR(f'data/cifar{nclass}',
#                    transform=transform, target_transform=one_hot(int(nclass)),
#                    train=True, download=True)
#     traind = IndexDatasetWrapper(traind)
#     testd = CIFAR(f'data/cifar{nclass}',
#                   transform=transform, target_transform=one_hot(int(nclass)),
#                   train=False, download=True)
#     testd = IndexDatasetWrapper(testd)

#     if ep == 2:  # using orig train and test
#         if fn == 'test.txt':
#             return testd
#         else:  # train.txt and database.txt
#             return traind

#     combine_data = np.concatenate([traind.data, testd.data], axis=0)
#     combine_targets = np.concatenate([traind.targets, testd.targets], axis=0)

#     path = f'data/cifar{nclass}/0_0_{ep}_{fn}'

#     load_data = fn == 'train.txt'
#     load_data = load_data and (reset or not os.path.exists(path))

#     if not load_data:
#         logging.info(f'Loading {path}')
#         data_index = torch.load(path)
#     else:
#         train_data_index = []
#         query_data_index = []
#         db_data_index = []

#         data_id = np.arange(combine_data.shape[0])  # [0, 1, ...]

#         for i in range(nclass):
#             class_mask = combine_targets == i
#             index_of_class = data_id[class_mask].copy()  # index of the class [2, 10, 656,...]
#             np.random.shuffle(index_of_class)

#             if ep == 1:
#                 query_n = 100  # // (nclass // 10)
#                 train_n = 500  # // (nclass // 10)

#                 index_for_query = index_of_class[:query_n].tolist()

#                 index_for_db = index_of_class[query_n:].tolist()
#                 index_for_train = index_for_db[:train_n]
#             elif ep == 2:  # ep2 = take all data
#                 query_n = 1000  # // (nclass // 10)

#                 index_for_query = index_of_class[:query_n].tolist()
#                 index_for_db = index_of_class[query_n:].tolist()
#                 index_for_train = index_for_db

#             elif ep == 3:  # Bi-Half Cifar10(II)
#                 query_n = 1000
#                 train_n = 500
#                 index_for_query = index_of_class[:query_n].tolist()
#                 index_for_db = index_of_class[query_n:].tolist()
#                 index_for_train = index_for_db[:train_n]

#             else:
#                 raise NotImplementedError('')

#             train_data_index.extend(index_for_train)
#             query_data_index.extend(index_for_query)
#             db_data_index.extend(index_for_db)

#         train_data_index = np.array(train_data_index)
#         query_data_index = np.array(query_data_index)
#         db_data_index = np.array(db_data_index)

#         torch.save(train_data_index, f'data/cifar{nclass}/0_0_{ep}_train.txt')
#         torch.save(query_data_index, f'data/cifar{nclass}/0_0_{ep}_test.txt')
#         torch.save(db_data_index, f'data/cifar{nclass}/0_0_{ep}_database.txt')

#         data_index = {
#             'train.txt': train_data_index,
#             'test.txt': query_data_index,
#             'database.txt': db_data_index
#         }[fn]

#     traind.data = combine_data[data_index]
#     traind.targets = combine_targets[data_index]

#     return traind


def cifar(nclass, **kwargs):
    transform = kwargs['transform']
    ep = kwargs['evaluation_protocol']
    fn = kwargs['filename']
    reset = kwargs['reset']

    prefix = DATA_FOLDER['cifar']

    CIFAR = CIFAR10 if int(nclass) == 10 else CIFAR100
    traind = CIFAR(f'{prefix}{nclass}',
                   transform=transform, target_transform=one_hot(int(nclass)),
                   train=True, download=True)
    traind = IndexDatasetWrapper(traind)
    testd = CIFAR(f'{prefix}{nclass}', train=False, download=True)
    testd = IndexDatasetWrapper(testd)

    combine_data = np.concatenate([traind.data, testd.data], axis=0)
    combine_targets = np.concatenate([traind.targets, testd.targets], axis=0)

    path = f'{prefix}{nclass}/0_{ep}_{fn}'

    load_data = fn == 'train.txt'
    load_data = load_data and (reset or not os.path.exists(path))


    if not load_data:
        print(f'Loading {path}')
        data_index = torch.load(path)
    else:
        train_data_index = []
        query_data_index = []
        db_data_index = []

        data_id = np.arange(combine_data.shape[0])  # [0, 1, ...]

        ### 22零样本划分
        class_ids = [x for x in range(nclass)]
        random.shuffle(class_ids)
        seen_class = class_ids[:8]
        unseen_class = class_ids[8:]
        for i in seen_class:
            class_mask = combine_targets == i
            index_of_class = data_id[class_mask].copy()  # index of the class [2, 10, 656,...]
            np.random.shuffle(index_of_class)

            train_n = 500  # // (nclass // 10)

            index_for_train = index_of_class[:train_n].tolist()
            index_for_db = index_of_class[train_n:].tolist()

            train_data_index.extend(index_for_train)
            db_data_index.extend(index_for_db)
        for i in unseen_class:
            class_mask = combine_targets == i
            index_of_class = data_id[class_mask].copy()  # index of the class [2, 10, 656,...]
            np.random.shuffle(index_of_class)

            query_n = 100  # // (nclass // 10)

            index_for_query = index_of_class[:query_n].tolist()
            index_for_db = index_of_class[query_n:].tolist()

            query_data_index.extend(index_for_query)
            db_data_index.extend(index_for_db)

        ### 21原始划分：每类数据打乱，然后取前1000作为test query，后面的作为db
        # for i in range(nclass):
        #     class_mask = combine_targets == i
        #     index_of_class = data_id[class_mask].copy()  # index of the class [2, 10, 656,...]
        #     np.random.shuffle(index_of_class)

        #     if ep == 1:
        #         #21论文原始划分
        #         query_n = 100  # // (nclass // 10)
        #         train_n = 500  # // (nclass // 10)

        #         index_for_query = index_of_class[:query_n].tolist()

        #         index_for_db = index_of_class[query_n:].tolist()
        #         index_for_train = index_for_db[:train_n]
        #     else:  # ep2 = take all data
        #         query_n = 1000  # // (nclass // 10)

        #         index_for_query = index_of_class[:query_n].tolist()
        #         index_for_db = index_of_class[query_n:].tolist()
        #         index_for_train = index_for_db
        #         print(i, len(index_for_query), len(index_for_db), len(index_for_train))

        #     train_data_index.extend(index_for_train)
        #     query_data_index.extend(index_for_query)
        #     db_data_index.extend(index_for_db)

 
        train_data_index = np.array(train_data_index)
        query_data_index = np.array(query_data_index)
        db_data_index = np.array(db_data_index)

        torch.save(train_data_index, f'data/cifar{nclass}/0_{ep}_train.txt')
        torch.save(query_data_index, f'data/cifar{nclass}/0_{ep}_test.txt')
        torch.save(db_data_index, f'data/cifar{nclass}/0_{ep}_database.txt')

        data_index = {
            'train.txt': train_data_index,
            'test.txt': query_data_index,
            'database.txt': db_data_index
        }[fn]
        # print(len(train_data_index)) #4000    5000
        # print(len(query_data_index)) #200     1000
        # print(len(db_data_index))    #55800   59000   1000测试集和数据库应该是隔离的，5000训练数据应该是从59000里选的
        # bb
    #print(len(combine_data),combine_data.shape) #60000 (60000, 32, 32, 3)
    
    traind.data = combine_data[data_index]
    traind.targets = combine_targets[data_index]
    #print(combine_targets.shape)  #60000  即0-9的十类类别标签
    #print(min(combine_targets))  0 

    return traind

def awa2(**kwargs):
    transform = kwargs['transform']
    filename = kwargs['filename']

    d = AwA2Dataset(DATA_FOLDER['awa2'], transform=transform, filename=filename)
    return d

def cub(**kwargs):
    transform = kwargs['transform']
    filename = kwargs['filename']

    d = CubDataset(DATA_FOLDER['cub'], transform=transform, filename=filename)
    return d

def sun(**kwargs):
    transform = kwargs['transform']
    filename = kwargs['filename']

    d = SunDataset(DATA_FOLDER['sun'], transform=transform, filename=filename)
    return d
    
def imagenet100(**kwargs):
    transform = kwargs['transform']
    filename = kwargs['filename']
    suffix = kwargs.get('dataset_name_suffix', '')

    d = HashingDataset(f'data/imagenet{suffix}', transform=transform, filename=filename, ratio=kwargs.get('ratio', 1))
    return d


def cars(**kwargs):
    transform = kwargs['transform']
    filename = kwargs['filename']

    d = HashingDataset('data/cars', transform=transform, filename=filename, ratio=kwargs.get('ratio', 1))
    return d


def landmark(**kwargs):
    transform = kwargs['transform']
    filename = kwargs['filename']
    return_id = kwargs.get('return_id', False)

    d = LandmarkDataset('data/landmark', transform=transform, filename=filename, return_id=return_id)
    return d


def nuswide(**kwargs):
    transform = kwargs['transform']
    filename = kwargs['filename']
    separate_multiclass = kwargs.get('separate_multiclass', False)
    suffix = kwargs.get('dataset_name_suffix', '')

    d = HashingDataset(f'data/nuswide_v2_256{suffix}',
                       transform=transform,
                       filename=filename,
                       separate_multiclass=separate_multiclass,
                       ratio=kwargs.get('ratio', 1))
    return d


def nuswide_single(**kwargs):
    return nuswide(separate_multiclass=True, **kwargs)


def coco(**kwargs):
    transform = kwargs['transform']
    filename = kwargs['filename']
    suffix = kwargs.get('dataset_name_suffix', '')

    d = HashingDataset(f'data/coco{suffix}', transform=transform, filename=filename, ratio=kwargs.get('ratio', 1))
    return d


def roxford5k(**kwargs):
    transform = kwargs['transform']
    filename = kwargs['filename']
    d = ROxfordParisDataset(dataset='roxford5k', filename=filename, transform=transform)
    return d


def rparis6k(**kwargs):
    transform = kwargs['transform']
    filename = kwargs['filename']
    d = ROxfordParisDataset(dataset='rparis6k', filename=filename, transform=transform)
    return d


def gldv2delgembed(**kwargs):
    filename = kwargs['filename']
    d = EmbeddingDataset('data/gldv2delgembed', filename=filename)
    return d


def roxford5kdelgembed(**kwargs):
    filename = kwargs['filename']
    d = EmbeddingDataset('data/roxford5kdelgembed', filename=filename)
    return d


def rparis6kdelgembed(**kwargs):
    filename = kwargs['filename']
    d = EmbeddingDataset('data/rparis6kdelgembed', filename=filename)
    return d


def descriptor(**kwargs):
    filename = kwargs['filename']
    data_folder = kwargs['data_folder']
    d = DescriptorDataset(data_folder, filename=filename, ratio=kwargs.get('ratio', 1))
    return d


def mirflickr(**kwargs):
    transform = kwargs['transform']
    filename = kwargs['filename']
    suffix = kwargs.get('dataset_name_suffix', '')

    d = HashingDataset(f'data/mirflickr{suffix}', transform=transform, filename=filename, ratio=kwargs.get('ratio', 1))
    return d


def sop_instance(**kwargs):
    transform = kwargs['transform']
    filename = kwargs['filename']

    d = SingleIDDataset('data/sop_instance', transform=transform, filename=filename)
    return d


def sop(**kwargs):
    transform = kwargs['transform']
    filename = kwargs['filename']
    suffix = kwargs.get('dataset_name_suffix', '')

    d = HashingDataset(f'data/sop{suffix}', transform=transform, filename=filename, ratio=kwargs.get('ratio', 1))
    return d


def food101(**kwargs):
    transform = kwargs['transform']
    filename = kwargs['filename']

    d = HashingDataset('data/food-101', transform=transform, filename=filename, ratio=kwargs.get('ratio', 1))
    return d
