# -*- coding: utf-8 -*

# -------------------------------------------------------------------------------
# Author: LiuNing
# Contact: 2742229056@qq.com
# Software: PyCharm
# File: mydataset.py
# Time: 8/1/19 3:33 PM
# Description: 
# -------------------------------------------------------------------------------

from torch.utils.data import Dataset
import os
from PIL import Image
import random
import numpy as np


########################################################################
# return image and label
#
class dataset(Dataset):
    def __init__(self, root, label, flag=1, signal=' ', transform=None):
        self._root = root
        self._flag = flag
        self._label = label
        self._transform = transform
        self._signal = signal
        self._list_images(self._root, self._label, self._signal)

    def _list_images(self, root, label, signal):
        self.synsets = []
        self.synsets.append(root)
        self.items = []

        c = 0
        for line in label:
            cls = line.rstrip('\n').split(signal)
            fn = cls.pop(0)

            if os.path.isfile(os.path.join(root, fn)):
                self.items.append((os.path.join(root, fn), float(cls[0])))
            else:
                print(os.path.join(root, fn))
            c += 1
        print('the total image is ', c)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):

        img = Image.open(self.items[index][0])
        img = img.convert('RGB')
        label = self.items[index][1]
        if self._transform is not None:
            return self._transform(img), label
        return img, label


########################################################################
# return image, label, and direction(left/right)
#
class dataset_direction(Dataset):
    def __init__(self, root, label, flag=1, signal=' ', transform=None):
        self._root = root
        self._flag = flag
        self._label = label
        self._transform = transform
        self._signal = signal
        self._list_images(self._root, self._label, self._signal)

    def _list_images(self, root, label, signal):
        self.synsets = []
        self.synsets.append(root)
        self.items = []

        c = 0
        for line in label:
            cls = line.rstrip('\n').split(signal)
            fn = cls.pop(0)

            if os.path.isfile(os.path.join(root, fn)):
                self.items.append((os.path.join(root, fn), float(cls[0]), float(cls[1])))
            else:
                print(os.path.join(root, fn))
            c += 1
        print('the total image is ', c)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):

        img = Image.open(self.items[index][0])
        img = img.convert('RGB')
        label = self.items[index][1]
        direction = self.items[index][2]
        if self._transform is not None:
            return self._transform(img), label, direction
        return img, label, direction


########################################################################
# triplet load data, return anchor,positive and negtive
#
class dataset_direction_triplet(Dataset):
    def __init__(self, root, label, flag=1, signal=' ', transform=None):
        self._root = root
        self._flag = flag
        self._label = label
        self._transform = transform
        self._signal = signal
        self._list_images(self._root, self._label, self._signal)
        self._dict_train = self.get_train_dict()
        self._labels = self.get_train_label()

    def _list_images(self, root, label, signal):
        self.synsets = []
        self.synsets.append(root)
        self.items = []

        c = 0
        for line in label:
            cls = line.rstrip('\n').split(signal)
            fn = cls.pop(0)

            if os.path.isfile(os.path.join(root, fn)):
                self.items.append((os.path.join(root, fn), float(cls[0]), float(cls[1])))
            else:
                print(os.path.join(root, fn))
            c += 1
        print('the total image is ', c)

    def get_train_label(self):
        labels = []
        for name, label, direct in self.items:
            labels.append(label)
        return labels

    def get_train_dict(self):
        dict_train = {}
        for name, label, direct in self.items:
            if not label in dict_train.keys():
                dict_train[label] = [(name, direct)]
            else:
                dict_train[label].append((name, direct))
        return dict_train

    def get_image(self, image_name, transformer):
        img = Image.open(image_name)
        img = img.convert('RGB')
        if transformer is not None:
            return transformer(img)
        return img

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):

        anchor_name = self.items[index][0]
        anchor_label = self.items[index][1]
        anchor_direct = self.items[index][2]
        names = self._dict_train[anchor_label]
        nums = len(names)
        assert nums > 2

        positive_name, positive_direct = random.choice(list(set(names) ^ set([(anchor_name, anchor_direct)])))
        negative_label = random.choice(list(set(self._labels) ^ set([anchor_label])))
        negative_name, negative_direct = random.choice(self._dict_train[negative_label])

        anchor_image = self.get_image(anchor_name, self._transform)
        positive_image = self.get_image(positive_name, self._transform)
        negative_image = self.get_image(negative_name, self._transform)

        assert negative_name != anchor_name

        return [anchor_image, positive_image, negative_image], \
               [anchor_label, anchor_label, negative_label], \
               [anchor_direct, positive_direct, negative_direct]


########################################################################
# load test, return image, image_name
#
class dataset_unlabeled(Dataset):
    def __init__(self, root, label, flag=1, transform=None):
        self._root = root
        self._flag = flag
        self._label = label
        self._transform = transform
        self._list_images(self._root, self._label)

    def _list_images(self, root, image_names):
        self.synsets = []
        self.synsets.append(root)
        self.items = []

        c = 0
        for line in image_names:
            image_name = line.rstrip('\n')

            if os.path.isfile(os.path.join(root, image_name)):
                self.items.append((os.path.join(root, image_name), image_name))
            else:
                print(os.path.join(root, image_name))
            c += 1
        print('the total image is ', c)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):

        img = Image.open(self.items[index][0])
        img = img.convert('RGB')
        image_name = self.items[index][1]
        if self._transform is not None:
            return self._transform(img), image_name
        return img, image_name
