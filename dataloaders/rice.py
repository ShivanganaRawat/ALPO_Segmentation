# Originally written by Kazuto Nakashima 
# https://github.com/kazuto1011/deeplab-pytorch

from base import BaseDataSet, BaseDataLoader
from utils import palette
import numpy as np
import os
import scipy
import torch
from PIL import Image
import cv2
from torch.utils.data import Dataset
from torchvision import transforms
import csv
from .randaugment import RandAugmentPolicy


class RiceDataset(BaseDataSet):
    """
    Contrains both SBD and VOC 2012 dataset
    Annotations : https://github.com/DrSleep/tensorflow-deeplab-resnet#evaluation
    Image Sets: https://ucla.app.box.com/s/rd9z2xvwsfpksi7mi08i2xqrj7ab4keb/file/55053033642
    """
    def __init__(self, create_dataset_using_txt=False, load_from=None, num_policies=None, magnitude=None, max_param=None, randaug=True, resize=0.0,**kwargs):
        self.num_classes = 2
        self.palette = [0, 0, 0, 255, 0, 0]
        self.create_dataset_using_txt = create_dataset_using_txt
        self.load_from = load_from
        self.num_policies = num_policies
        self.magnitude = magnitude
        self.max_param = max_param
        self.randaug = randaug
        self.resize = resize
        super(RiceDataset, self).__init__(**kwargs)

    def _set_files(self):

        image_dir = os.path.join(self.root, "images")
        label_dir = os.path.join(self.root, "labels")

        if self.create_dataset_using_txt:
            file_list_path = self.load_from
        else:
            file_list_path = os.path.join(self.root, self.split + ".csv")
        file_set = csv.reader(open(file_list_path, 'rt'))
        file_list = [r[0] for r in file_set]
        
        self.image_path, self.label_path = [], []
        for file in file_list:
            #folder = file.split('_')[0] + '_' + file.split('_')[1]
            self.image_path.append(os.path.join(image_dir, file+".jpg"))
            self.label_path.append(os.path.join(label_dir, file+"_label.png"))
        self.image_path.sort()
        self.label_path.sort()
        self.files = list(zip(self.image_path, self.label_path))
    
    def _load_data(self, index):
        image_path = self.files[index][0]
        label_path = self.files[index][1]
        #print(image_path)
        #print(label_path)
        image = Image.open(image_path)
        if self.randaug:
            image = self.rand_aug(image)
        image = np.asarray(image, dtype=np.uint8)
        label = np.asarray(Image.open(label_path), dtype=np.uint8)
        if self.resize:
            image = cv2.resize(image, (self.crop_w, self.crop_h), interpolation=cv2.INTER_LINEAR)
            label = cv2.resize(label, (self.crop_w, self.crop_h), interpolation=cv2.INTER_NEAREST)
        assert label.shape[0] == image.shape[0] and label.shape[1] == image.shape[1]
        image_id = os.path.splitext(os.path.basename(self.files[index][0]))[0]
        image = image.astype(np.float32)
        label = label.astype(np.int32)
        #print(self.split,image_path)
        #print(self.split,label.shape)
        return image, label, image_id


class Rice(BaseDataLoader):
    def __init__(self, data_dir, batch_size, split, crop_size=None, base_size=None, scale=True, num_workers=1, val=False,
                    shuffle=False, flip=False, rotate=False, blur= False, augment=False, val_split= None, return_id=False, 
                    create_dataset_using_txt=False, load_from=None, crop_h=None, crop_w=None, num_policies=None, magnitude=None, max_param=None, randaug=False, hflip=False, resize=False):
        
        self.MEAN = [0.40107516, 0.55142987, 0.35571281]
        self.STD = [0.19988729, 0.23059717, 0.18807018]
        
        #self.MEAN = [0.40117231, 0.55175546, 0.35593277]
        #self.STD = [0.19970677, 0.23046128, 0.18788528]
        
        #self.MEAN = [0.40095165, 0.55172258, 0.35559890]
        #self.STD = [0.20249349, 0.23500672, 0.19094837]

        kwargs = {
            'root': data_dir,
            'split': split,
            'mean': self.MEAN,
            'std': self.STD,
            'augment': augment,
            'crop_size': crop_size,
            'crop_h': crop_h,
            'crop_w': crop_w,
            'base_size': base_size,
            'scale': scale,
            'flip': flip,
            'blur': blur,
            'rotate': rotate,
            'return_id': return_id,
            'val': val,
            'hflip':hflip
        }
        self.dataset = RiceDataset(create_dataset_using_txt=create_dataset_using_txt, load_from=load_from, num_policies=num_policies, magnitude=magnitude, max_param=max_param, randaug=randaug, resize=resize,**kwargs)
        super(Rice, self).__init__(self.dataset, batch_size, shuffle, num_workers, val_split)
