# -*- coding: utf-8 -*

# -------------------------------------------------------------------------------
# Author: LiuNing
# Contact: 2742229056@qq.com
# Software: PyCharm
# File: model.py
# Time: 6/26/19 6:57 PM
# Description: 
# -------------------------------------------------------------------------------

from core import *
from core.utils import *
from core.seed import *
from block.senet import seresnet50


########################################################################
# Solution One: tiger_cnn1
# Multitask: Tiger ID, Left/right
# Backbone: SE-ResNet50
# Loss Function: LabelSmoothingCrossEntropy
#
class tiger_cnn1(nn.Module):
    def __init__(self, classes=107, stride=1):
        super(tiger_cnn1, self).__init__()
        self.classes = classes
        model = seresnet50(pretrained=True)
        if stride == 1:
            model.layer4[0].downsample[0].stride = (1, 1)
            model.layer4[0].conv2.stride = (1, 1)
        self.backbone = model

        self.fc7 = nn.Sequential(
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5)
        )
        self.cls = nn.Linear(512, classes)
        self.cls_direction = nn.Linear(512, 2)

        self.loss = LabelSmoothingCrossEntropy(smoothing=0.1)

    def fix_params(self, is_training=True):
        for p in self.backbone.parameters():
            p.requires_grad = is_training

    def get_loss(self, logits, labels, direction):
        loss = self.loss(logits[0], labels) + self.loss(logits[1], direction)
        return loss

    def features(self, x):
        # backbone
        x = self.backbone.layer0(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        x = torch.mean(torch.mean(x, dim=2), dim=2)
        x = self.fc7(x)

        return [x, ]

    def forward(self, x, label=None, direction=None):
        # backbone
        self.image = x
        x = self.backbone.layer0(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        x = torch.mean(torch.mean(x, dim=2), dim=2)
        x = self.fc7(x)

        # TigerID
        glogit = self.cls(x)
        # Left/Right
        dlogit = self.cls_direction(x)

        return [glogit, dlogit]


########################################################################
# Solution Two: tiger_cnn2
# Multitask: Tiger ID, Left/Right, IsTiger/NoTiger
# Backbone: SE-ResNet50
# Loss Function: LabelSmoothingCrossEntropy
#
class tiger_cnn2(nn.Module):
    def __init__(self, classes=107, stride=1):
        super(tiger_cnn2, self).__init__()
        self.classes = classes

        model = seresnet50(pretrained=True)
        if stride == 1:
            model.layer4[0].downsample[0].stride = (1, 1)
            model.layer4[0].conv2.stride = (1, 1)
        self.backbone = model

        self.fc7 = nn.Sequential(
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5)
        )
        self.cls = nn.Linear(512, classes)
        self.cls_direction = nn.Linear(512, 2)
        self.cls_tiger = nn.Linear(512, 2)

        self.loss = LabelSmoothingCrossEntropy(smoothing=0.1)

    def fix_params(self, is_training=True):
        for p in self.backbone.parameters():
            p.requires_grad = is_training

    def get_loss(self, logits, labels, direction):

        b_size = labels.shape[0]
        loss2 = 0
        num = 0
        loss1 = 0
        for i in range(b_size):
            if labels[i] == -1:
                loss1 += self.loss(logits[2][i].unsqueeze(dim=0), (labels[i] + 2).unsqueeze(dim=0))
            else:
                num += 1
                loss1 += self.loss(logits[2][i].unsqueeze(dim=0), (labels[i] - labels[i]).unsqueeze(dim=0))
                loss2 += self.loss(logits[0][i].unsqueeze(dim=0), labels[i].unsqueeze(dim=0)) + \
                         self.loss(logits[1][i].unsqueeze(dim=0), direction[i].unsqueeze(dim=0))
        if num != 0:
            loss = loss1 / b_size + 2 * loss2 / num
        else:
            loss = 0
        return loss

    def features(self, x):
        # backbone
        x = self.backbone.layer0(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        x = torch.mean(torch.mean(x, dim=2), dim=2)
        x = self.fc7(x)

        return [x, ]

    def forward(self, x, label=None, direction=None):
        # backbone
        self.image = x

        x = self.backbone.layer0(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        x = torch.mean(torch.mean(x, dim=2), dim=2)
        x = self.fc7(x)

        # TigerID
        glogit = self.cls(x)
        # Left/Right
        dlogit = self.cls_direction(x)
        # IsTiger/NoTiger
        tlogit = self.cls_tiger(x)

        return [glogit, dlogit, tlogit]


########################################################################
# Solution Three: tiger_cnn3
# Multitask: Tiger ID, Left/right
# Backbone: SE-ResNet50
# DoubleBranch: backbone, erase
# Loss Function: LabelSmoothingCrossEntropy
#
class tiger_cnn3(nn.Module):
    def __init__(self, classes=107, stride=1):
        super(tiger_cnn3, self).__init__()
        self.classes = classes

        # backbone
        model = seresnet50(pretrained=True)
        if stride == 1:
            model.layer4[0].downsample[0].stride = (1, 1)
            model.layer4[0].conv2.stride = (1, 1)
        self.backbone = model
        self.fc7 = nn.Sequential(
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5)
        )
        self.cls = nn.Linear(512, classes)
        self.cls_direction = nn.Linear(512, 2)

        # erase
        model2 = seresnet50(pretrained=True)
        if stride == 1:
            model2.layer4[0].downsample[0].stride = (1, 1)
            model2.layer4[0].conv2.stride = (1, 1)
        self.erase = model2
        self.erase_fc7 = nn.Sequential(
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5)
        )
        self.erase_cls = nn.Linear(512, classes)
        self.erase_cls_direction = nn.Linear(512, 2)

        # fuse
        self.fuse_fc7 = nn.Sequential(
            nn.Linear(4096, 512),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5)
        )
        self.fuse_cls = nn.Linear(512, classes)
        self.fuse_cls_direction = nn.Linear(512, 2)

        self.my_upsample = nn.Upsample(size=(288, 448), mode='bilinear', align_corners=True)
        self.loss = LabelSmoothingCrossEntropy(smoothing=0.1)

    def normalize_atten_maps(self, atten_maps):
        atten_shape = atten_maps.size()
        batch_mins, _ = torch.min(atten_maps.view(atten_shape[0:-2] + (-1,)), dim=-1, keepdim=True)
        batch_maxs, _ = torch.max(atten_maps.view(atten_shape[0:-2] + (-1,)), dim=-1, keepdim=True)
        atten_normed = (atten_maps.view(atten_shape[0:-2] + (-1,)) - batch_mins) / (batch_maxs - batch_mins)
        atten_normed = atten_normed.view(atten_shape)
        return atten_normed

    def erase_feature_maps(self, atten_map_normed, feature_maps, threshold, flag=False):
        if len(atten_map_normed.size()) > 3:
            atten_map_normed = torch.squeeze(atten_map_normed)
        atten_shape = atten_map_normed.size()

        if flag:
            mask = torch.zeros_like(atten_map_normed)
            for i in range(atten_shape[0]):
                tmp = torch.zeros_like(atten_map_normed[i]).cuda()
                pos = torch.ge(atten_map_normed[i], threshold)
                tmp[pos.data] = 1.0
                mask[i] = tmp
        else:
            mask = torch.ones_like(atten_map_normed)
            for i in range(atten_shape[0]):
                tmp = torch.ones_like(atten_map_normed[i])
                pos = torch.ge(atten_map_normed[i], threshold)
                tmp[pos.data] = 0.0
                mask[i] = tmp
        mask = torch.unsqueeze(mask, dim=1)

        erased_feature_maps = feature_maps * Variable(mask)
        return erased_feature_maps

    def fix_params(self, is_training=True):
        for p in self.backbone.parameters():
            p.requires_grad = is_training
        for p in self.erase.parameters():
            p.requires_grad = is_training

    def get_loss(self, logits, labels, direction):

        loss1 = self.loss(logits[0], labels) + self.loss(logits[1], direction)
        loss2 = self.loss(logits[2], labels) + self.loss(logits[3], direction)
        loss3 = self.loss(logits[4], labels) + self.loss(logits[5], direction)
        loss = loss1 + loss2 + loss3
        return loss

    def features(self, x):
        # backbone
        x1 = x.detach()
        x = self.backbone.layer0(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        x = torch.mean(torch.mean(x, dim=2), dim=2)

        # erase
        x1 = self.erase.layer0(x1)
        x1 = self.erase.layer1(x1)
        x1 = self.erase.layer2(x1)
        x1 = self.erase.layer3(x1)
        x1 = self.erase.layer4(x1)

        x1 = torch.mean(torch.mean(x1, dim=2), dim=2)

        # fuse
        fuse_f = torch.cat((x, x1), dim=1)
        x2 = self.fuse_fc7(fuse_f)

        return [x2, ]

    def forward(self, x, label=None, direction=None):
        # backbone
        self.image = x.detach()
        x = self.backbone.layer0(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        atten_map = torch.sum(x.detach(), dim=1)
        atten_map = self.my_upsample(self.normalize_atten_maps(atten_map).unsqueeze(dim=1))

        x = torch.mean(torch.mean(x, dim=2), dim=2)
        backbone_f = x.detach()
        x = self.fc7(x)

        # TigerID
        glogit = self.cls(x)
        # Left/Right
        dlogit = self.cls_direction(x)

        # erase
        x1 = self.erase_feature_maps(atten_map.detach(), self.image, threshold=0.6, flag=False)
        x1 = self.erase.layer0(x1)
        x1 = self.erase.layer1(x1)
        x1 = self.erase.layer2(x1)
        x1 = self.erase.layer3(x1)
        x1 = self.erase.layer4(x1)

        x1 = torch.mean(torch.mean(x1, dim=2), dim=2)
        erase_f = x1.detach()
        x1 = self.erase_fc7(x1)

        # TigerID
        erase_glogit = self.erase_cls(x1)
        # Left/Right
        erase_dlogit = self.erase_cls_direction(x1)

        # fuse
        fuse_f = torch.cat((backbone_f, erase_f), dim=1)
        x2 = self.fuse_fc7(fuse_f)

        # TigerID
        fuse_glogit = self.fuse_cls(x2)
        # Left/Right
        fuse_dlogit = self.fuse_cls_direction(x2)

        return [glogit, dlogit, erase_glogit, erase_dlogit, fuse_glogit, fuse_dlogit]


########################################################################
# Solution Four: tiger_cnn4
# Multitask: Tiger ID, Left/right
# Backbone: SE-ResNet50
# Loss Function: LabelSmoothingCrossEntropy+TripletLoss
#
class tiger_cnn4(nn.Module):
    def __init__(self, classes=107, stride=1):
        super(tiger_cnn4, self).__init__()
        self.classes = classes

        model = seresnet50(pretrained=True)
        if stride == 1:
            model.layer4[0].downsample[0].stride = (1, 1)
            model.layer4[0].conv2.stride = (1, 1)
        self.backbone = model

        self.fc7 = nn.Sequential(
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
        )
        self.cls = nn.Linear(512, classes)
        self.cls_direction = nn.Linear(512, 2)

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.loss = LabelSmoothingCrossEntropy(smoothing=0.1)

    def fix_params(self, is_training=True):
        for p in self.backbone.parameters():
            p.requires_grad = is_training

    def get_loss(self, logits, labels, direction):

        loss = self.loss(logits[0], labels) + self.loss(logits[1], direction)
        triplet_loss = global_loss(TripletLoss(margin=0.3), logits[2], labels)[0]
        return loss + triplet_loss

    def features(self, x):
        # backbone
        x = self.backbone.layer0(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        x = torch.mean(torch.mean(x, dim=2), dim=2)
        x = self.fc7(x)
        x = l2_norm(x)

        return [x, ]

    def forward(self, x, label=None, direction=None):
        # backbone
        self.image = x
        x = self.backbone.layer0(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        x = torch.mean(torch.mean(x, dim=2), dim=2)
        x = self.fc7(x)
        x = l2_norm(x)

        # TigerID
        glogit = self.cls(x)
        # Left/Right
        dlogit = self.cls_direction(x)

        return [glogit, dlogit, x]


########################################################################
# Solution Five: tiger_cnn5
# Multitask: Tiger ID, Left/right
# Model Fusion: tiger_cnn1, tiger_cnn3
# Loss Function: LabelSmoothingCrossEntropy+TripletLoss
#
class tiger_cnn5(nn.Module):
    def __init__(self, classes=107):
        super(tiger_cnn5, self).__init__()

        model1 = tiger_cnn1(classes)
        for p in model1.parameters():
            p.requires_grad = False
        self.model1 = model1

        model2 = tiger_cnn3(classes)
        for p in model2.parameters():
            p.requires_grad = False
        self.model2 = model2

        self.fc7 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
        )
        self.cls = nn.Linear(512, classes)
        self.cls_direction = nn.Linear(512, 2)
        self.loss = LabelSmoothingCrossEntropy(smoothing=0.1)

    def get_loss(self, logits, labels, direction):
        loss = self.loss(logits[0], labels) + self.loss(logits[1], direction)
        return loss

    def features(self, x):
        # backbone
        x1 = self.model1.features(x)[0]
        x2 = self.model2.features(x)[0]

        x = torch.cat((x1, x2), dim=1)
        x = self.fc7(x)

        return [x, ]

    def forward(self, x, label=None, direction=None):

        # backbone
        x1 = self.model1.features(x)[0]
        x2 = self.model2.features(x)[0]

        x = torch.cat((x1, x2), dim=1)
        x = self.fc7(x)

        # TigerID
        glogit = self.cls(x)
        # Left/Right
        dlogit = self.cls_direction(x)

        return [glogit, dlogit]


########################################################################
# Solution Five: tiger_cnn6
# Multitask: Tiger ID, Left/right
# Model Fusion: tiger_cnn1, tiger_cnn2
# Loss Function: LabelSmoothingCrossEntropy+TripletLoss
#
class tiger_cnn6(nn.Module):
    def __init__(self, classes=107):
        super(tiger_cnn6, self).__init__()

        model1 = tiger_cnn1(classes)
        for p in model1.parameters():
            p.requires_grad = False

        self.model1 = model1

        model2 = tiger_cnn2(classes)
        for p in model2.parameters():
            p.requires_grad = False
        self.model2 = model2

        self.fc7 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
        )
        self.cls = nn.Linear(512, classes)
        self.cls_direction = nn.Linear(512, 2)
        self.loss = LabelSmoothingCrossEntropy(smoothing=0.1)

    def get_loss(self, logits, labels, direction):
        loss = self.loss(logits[0], labels) + self.loss(logits[1], direction)
        return loss

    def features(self, x):
        # backbone
        x1 = self.model1.features(x)[0]
        x2 = self.model2.features(x)[0]

        x = torch.cat((x1, x2), dim=1)
        x = self.fc7(x)

        return [x, ]

    def forward(self, x, label=None, direction=None):

        # backbone
        x1 = self.model1.features(x)[0]
        x2 = self.model2.features(x)[0]

        x = torch.cat((x1, x2), dim=1)
        x = self.fc7(x)

        # TigerID
        glogit = self.cls(x)
        # Left/Right
        dlogit = self.cls_direction(x)

        return [glogit, dlogit]

########################################################################
# 测试函数是否正确
#
# if __name__ == '__main__':
#     net = tiger_cnn1(num_classes=10)
#     x = Variable(torch.randn(2, 3, 448, 448))
#     y = Variable(torch.ones(2, ).long())
#     print(net(x, y))
