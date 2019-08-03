# -*- coding: utf-8 -*

# -------------------------------------------------------------------------------
# Author: LiuNing
# Contact: 2742229056@qq.com
# Software: PyCharm
# File: dataloader.py
# Time: 7/27/19 1:17 PM
# Description: 
# -------------------------------------------------------------------------------


from torch.utils.data import DataLoader
from torchvision import transforms as transforms
from .preprocess import *
from .constant import *
from .mixup import *
from .mydataset import *


########################################################################
# read file
#
def get_label(label_path):
    f = open(label_path)
    lines = f.readlines()
    return lines


########################################################################
# load reid: train gallery probe
#
def load_data_gallery_probe(root, train_paths, gallery_paths, probe_paths, collate_fn=None, signal=' ',
                            resize_size=(256, 512), input_size=(224, 448), batch_size=32, num_workers=0):

    train_list = []
    for i in train_paths:
        tmp = get_label(i)
        train_list = train_list + tmp

    gallery_list = []
    for i in gallery_paths:
        tmp = get_label(i)
        gallery_list = gallery_list + tmp

    probe_list = []
    for i in probe_paths:
        tmp = get_label(i)
        probe_list = probe_list + tmp

    train_transformer = transforms.Compose([
        transforms.Resize(resize_size, Image.BILINEAR),
        transforms.RandomCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
        RandomErasing(),
        Cutout(),
    ])
    gallery_transformer = transforms.Compose([
        transforms.Resize(resize_size, Image.BILINEAR),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
    ])
    probe_transformer = transforms.Compose([
        transforms.Resize(resize_size, Image.BILINEAR),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
    ])

    train_dataset = dataset(root, train_list, flag=1, signal=signal, transform=train_transformer)
    gallery_dataset = dataset(root, gallery_list, flag=1, signal=signal, transform=gallery_transformer)
    probe_dataset = dataset(root, probe_list, flag=1, signal=signal, transform=probe_transformer)

    if collate_fn is None:
        collate_fn = torch.utils.data.dataloader.default_collate

    train_iter = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    gallery_iter = DataLoader(
        gallery_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    probe_iter = DataLoader(
        probe_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return train_iter, gallery_iter, probe_iter


########################################################################
# load reid: train gallery probe
#
def load_direction_gallery_probe(root, train_paths, gallery_paths, probe_paths, collate_fn=None, signal=' ',
                                 resize_size=(256, 512), input_size=(224, 448), batch_size=32, num_workers=0):

    train_list = []
    for i in train_paths:
        tmp = get_label(i)
        train_list = train_list + tmp

    gallery_list = []
    for i in gallery_paths:
        tmp = get_label(i)
        gallery_list = gallery_list + tmp

    probe_list = []
    for i in probe_paths:
        tmp = get_label(i)
        probe_list = probe_list + tmp

    train_transformer = transforms.Compose([
        transforms.Resize(resize_size, Image.BILINEAR),
        transforms.RandomCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
        RandomErasing(),
        Cutout(),
    ])
    gallery_transformer = transforms.Compose([
        transforms.Resize(resize_size, Image.BILINEAR),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
    ])
    probe_transformer = transforms.Compose([
        transforms.Resize(resize_size, Image.BILINEAR),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
    ])

    train_dataset = dataset_direction(root, train_list, flag=1, signal=signal, transform=train_transformer)
    gallery_dataset = dataset(root, gallery_list, flag=1, signal=signal, transform=gallery_transformer)
    probe_dataset = dataset(root, probe_list, flag=1, signal=signal, transform=probe_transformer)

    if collate_fn is None:
        collate_fn = torch.utils.data.dataloader.default_collate

    train_iter = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    gallery_iter = DataLoader(
        gallery_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    probe_iter = DataLoader(
        probe_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return train_iter, gallery_iter, probe_iter


########################################################################
# load reid: train gallery probe
#
def train_collate(batch):
    batch_size = len(batch)
    images = []
    labels = []
    directs = []
    for b in range(batch_size):
        if batch[b][0] is None:
            continue
        else:
            images.extend(batch[b][0])
            labels.extend(batch[b][1])
            directs.extend(batch[b][2])
    images = torch.stack(images, 0)
    labels = torch.from_numpy(np.array(labels))
    directs = torch.from_numpy(np.array(directs))
    return images, labels, directs


def load_triplet_direction_gallery_probe(root, train_paths, gallery_paths, probe_paths, collate_fn=None, signal=' ',
                                         resize_size=(256, 512), input_size=(224, 448), batch_size=32, num_workers=0):

    train_list = []
    for i in train_paths:
        tmp = get_label(i)
        train_list = train_list + tmp

    gallery_list = []
    for i in gallery_paths:
        tmp = get_label(i)
        gallery_list = gallery_list + tmp

    probe_list = []
    for i in probe_paths:
        tmp = get_label(i)
        probe_list = probe_list + tmp

    train_transformer = transforms.Compose([
        transforms.Resize(resize_size, Image.BILINEAR),
        transforms.RandomCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
        RandomErasing(),
        Cutout(),
    ])
    gallery_transformer = transforms.Compose([
        transforms.Resize(resize_size, Image.BILINEAR),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
    ])
    probe_transformer = transforms.Compose([
        transforms.Resize(resize_size, Image.BILINEAR),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
    ])

    train_dataset = dataset_direction_triplet(root, train_list, flag=1, signal=signal, transform=train_transformer)
    gallery_dataset = dataset(root, gallery_list, flag=1, signal=signal, transform=gallery_transformer)
    probe_dataset = dataset(root, probe_list, flag=1, signal=signal, transform=probe_transformer)

    if collate_fn is None:
        collate_fn = torch.utils.data.dataloader.default_collate

    train_iter = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    gallery_iter = DataLoader(
        gallery_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    probe_iter = DataLoader(
        probe_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return train_iter, gallery_iter, probe_iter


########################################################################
# load test
#
def load_unlabeled_data(root, test_paths, resize_size=(512, 512), input_size=(448, 448), batch_size=32,
                        num_workers=0):
    test_list = []
    for i in test_paths:
        tmp = get_label(i)
        test_list = test_list + tmp

    test_transformer = transforms.Compose([
        transforms.Resize(resize_size, Image.BILINEAR),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
    ])

    test_dataset = dataset_unlabeled(root, test_list, flag=1, transform=test_transformer)

    test_iter = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return test_iter

# if __name__ == '__main__':
#
#     train_paths = ['../datalist/dir/train.txt', ]
#     gallery_paths = ['../datalist/dir/gallery1.txt', ]
#     probe_paths = ['../datalist/dir/probe.txt', ]
#     train_iter, gallery_iter, probe_iter = load_triplet_direction_gallery_probe(
#         root='../database',
#         train_paths=train_paths,
#         gallery_paths=gallery_paths,
#         probe_paths=probe_paths,
#         signal=' ',
#         resize_size=(256, 384),
#         input_size=(256, 384),
#         batch_size=32,
#         num_workers=4,
#         collate_fn=train_collate
#     )
#
#     from tqdm import tqdm
#
#     for data in tqdm(train_iter, desc='Train'):
#         j = 4
#         inputs, labels, directs = data
#
#         print(directs)
#
#         img = inputs[j].cpu().data.numpy().transpose((1, 2, 0))
#         img = (img * IMAGENET_DEFAULT_STD + IMAGENET_DEFAULT_MEAN) * 255
#         img = np.array(img.astype('uint8'))
#
#         plt.imshow(img)
#         plt.show()
#
#         break
