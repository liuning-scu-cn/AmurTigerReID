# -*- coding: utf-8 -*

#-------------------------------------------------------------------------------
# Author: LiuNing
# Contact: 2742229056@qq.com
# Software: PyCharm
# File: test.py
# Time: 7/30/19 9:32 PM
# Description: test model
#-------------------------------------------------------------------------------

from torch.utils.data import Dataset, DataLoader
from PIL import Image
from dataload.constant import *
from torchvision import transforms
from core import *
from tqdm import tqdm

import json

init_environment()
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


########################################################################
# 读取label文件, 返回文件中所有的内容
#
def get_label(label_path):
    f = open(label_path)
    lines = f.readlines()
    return lines


########################################################################
# labeled dataset
#
class dataset(Dataset):
    def __init__(self, root, label, signal=' ', transform=None):
        self._root = root
        self._label = label
        self._signal = signal
        self._transform = transform
        self._list_images(self._root, self._label)

    def _list_images(self, root, image_names):
        self.synsets = []
        self.synsets.append(root)
        self.items = []

        c = 0
        for line in image_names:
            cls = line.rstrip('\n').split(self._signal)
            fn = cls.pop(0)

            if os.path.isfile(os.path.join(root, fn)):
                self.items.append((os.path.join(root, fn), fn, float(cls[0])))
            else:
                print(os.path.join(root, fn))
            c += 1
        print('the total image is ', c)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):

        img = Image.open(self.items[index][0])
        img = img.convert('RGB')
        image_name = self.items[index][1]
        label = self.items[index][2]
        if self._transform is not None:
            return self._transform(img), image_name, label
        return img, image_name, label


########################################################################
# unlabeled dataset
#
class dataset_unlabeled(Dataset):
    def __init__(self, root, label, transform=None):
        self._root = root
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


########################################################################
# dataloader
#
def load_gallery_probe_data(root, gallery_paths, probe_paths, resize_size=(324, 504), input_size=(288, 448),
                            signal=' ', batch_size=32, num_workers=0):
    gallery_list = []
    for i in gallery_paths:
        tmp = get_label(i)
        gallery_list = gallery_list + tmp

    probe_list = []
    for i in probe_paths:
        tmp = get_label(i)
        probe_list = probe_list + tmp

    transformer = transforms.Compose([
        transforms.Resize(resize_size, Image.BILINEAR),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
    ])

    gallery_dataset = dataset_unlabeled(root, gallery_list, transformer)
    probe_dataset = dataset_unlabeled(root, probe_list, transformer)

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

    return gallery_iter, probe_iter


def main():
    gallery_paths = ['./datalist/test.txt', ]
    probe_paths = ['./datalist/test.txt', ]

    gallery_iter, probe_iter = load_gallery_probe_data(
        root='/media/liuning/UBUNTU 16_0/ValidateTiger/database',
        gallery_paths=gallery_paths,
        probe_paths=probe_paths,
        signal=' ',
        resize_size=(324, 504),
        input_size=(288, 448),
        batch_size=16,
        num_workers=2
    )

    feature_size = 1024
    net = tiger_cnn5(classes=107)
    net.load_state_dict(torch.load(
        '/media/liuning/BRL_LiuNing/ICCVWorkshop/TigerChallenge/model/SEResNet50_TripletTiger_Finetuning_WarmUp_Direction_288_448_Gallery2_CutOut_RandomErase_FlipTest_0.001lr_10batchsize_20190731_111634/iter07_model.ckpt')['net_state_dict'])
    net = net.cuda()

    # val
    net.eval()
    gallery_features = []
    gallery_names = []
    query_features = []
    query_names = []
    for data in tqdm(gallery_iter, desc='Gallery'):
        with torch.no_grad():
            inputs, image_names = data
            b_size = inputs.size(0)

            ff = torch.FloatTensor(b_size, feature_size).zero_().cuda()

            flip_inputs = fliplr(inputs.detach())
            flip_inputs = Variable(flip_inputs.cuda())
            input_img = Variable(inputs.cuda())
            features = net.features(input_img)[0]
            flip_features = net.features(flip_inputs)[0]

            ff += torch.cat((features, flip_features), dim=1)
            fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
            ff = ff.div(fnorm.expand_as(ff))

            for i in range(b_size):
                gallery_features.append(ff[i].cpu().numpy())
                gallery_names.append(image_names[i])

    for data in tqdm(probe_iter, desc='Probe'):
        with torch.no_grad():
            inputs, image_names = data
            if inputs.size(0) == 1:
                continue
            b_size = inputs.size(0)

            ff = torch.FloatTensor(b_size, feature_size).zero_().cuda()

            flip_inputs = fliplr(inputs).detach()
            flip_inputs = Variable(flip_inputs.cuda())
            input_img = Variable(inputs.cuda())
            features = net.features(input_img)[0]
            flip_features = net.features(flip_inputs)[0]

            ff += torch.cat((features, flip_features), dim=1)
            fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
            ff = ff.div(fnorm.expand_as(ff))

            for i in range(b_size):
                query_features.append(ff[i].cpu().numpy())
                query_names.append(image_names[i])

    gallery_features = torch.FloatTensor(gallery_features)
    query_features = torch.FloatTensor(query_features)

    q_g_dist = np.dot(query_features, np.transpose(gallery_features))
    q_q_dist = np.dot(query_features, np.transpose(query_features))
    g_g_dist = np.dot(gallery_features, np.transpose(gallery_features))
    re_rank = re_ranking(q_g_dist, q_q_dist, g_g_dist)

    result = open('./result/result.json', 'w')
    my_result = []
    for i in range(len(query_names)):
        tmp = {}
        image_name = query_names[i].split('/')[1]
        index = np.argsort(re_rank[i, :])

        tmp['query_id'] = int(image_name.rstrip('.jpg'))
        p = 0
        gallery_tmp = []
        for j in index:
            if p == 0:
                p += 1
                continue
            current_name = gallery_names[j].split('/')[1]
            gallery_tmp.append(int(current_name.rstrip('.jpg')))
        tmp['ans_ids'] = gallery_tmp
        my_result.append(tmp)
    json.dump(my_result, result)

    print('finish')


if __name__ == '__main__':
    main()