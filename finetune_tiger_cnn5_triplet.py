# -*- coding: utf-8 -*

#-------------------------------------------------------------------------------
# Author: LiuNing
# Contact: 2742229056@qq.com
# Software: PyCharm
# File: finetune_triplet.py
# Time: 8/1/19 10:03 PM
# Description: 
#-------------------------------------------------------------------------------

import torch.optim as optim
from shutil import copyfile
from datetime import datetime
from tqdm import tqdm
from core import *
from dataload import *


init_environment()
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
multi_gpus = False
model_name = 'tiger_cnn5'


def main():
    save_dir = os.path.join(SAVE_DIR, model_name + '_' +
                            datetime.now().strftime('%Y%m%d_%H%M%S'))
    if os.path.exists(save_dir):
        raise NameError('model dir exists!')
    os.makedirs(save_dir)
    copyfile('./finetune_triplet.py.py', save_dir + '/train.py')
    copyfile('./core/model.py', save_dir + '/model.py')
    copyfile('./core/config.py', save_dir + '/config.py')
    logging = init_log(save_dir)
    _print = logging.info

    train_paths = ['./datalist/train.txt', ]
    gallery_paths = ['./datalist/gallery.txt', ]
    probe_paths = ['./datalist/probe.txt', ]

    train_iter, gallery_iter, probe_iter = load_direction_gallery_probe(
        root='./database',
        train_paths=train_paths,
        gallery_paths=gallery_paths,
        probe_paths=probe_paths,
        signal=' ',
        resize_size=RESIZE_SIZE,
        input_size=INPUT_SIZE,
        batch_size=BATCH_SIZE,
        num_workers=2
    )

    feature_size = 1024

    net = tiger_cnn5(classes=107)
    ignore_params = list(map(id, net.cls.parameters()))
    ignore_params += list(map(id, net.cls_direction.parameters()))
    ignore_params += list(map(id, net.fc7.parameters()))
    base_params = filter(lambda p: id(p) not in ignore_params, net.parameters())
    extra_params = filter(lambda p: id(p) in ignore_params, net.parameters())
    optimizer = optim.SGD(
        [{'params': base_params, 'lr': 0.001},
         {'params': extra_params, 'lr': 0.001}],
        weight_decay=1e-4, momentum=0.9, nesterov=True
    )

    exp_lr_scheduler = StepLRScheduler(optimizer=optimizer, decay_t=20, decay_rate=0.1, warmup_lr_init=1e-5, warmup_t=3)

    net.load_state_dict(torch.load('./model/tiger_cnn1/model.ckpt'))
    net.fix_params(is_training=False)
    net = net.cuda()
    if multi_gpus:
        net = nn.DataParallel(net).cuda()

    losses = AverageMeter()
    train_acc = AverageMeter()
    train_acc5 = AverageMeter()

    erase_train_acc = AverageMeter()

    max_test_acc = 0.0
    for epoch in range(TOTAL_EPOCH):

        # train
        net.train()
        flag = False
        exp_lr_scheduler.step(epoch)
        losses.reset()
        train_acc.reset()
        train_acc5.reset()
        erase_train_acc.reset()

        for data in tqdm(train_iter, desc='Train Epoch: {}'.format(epoch + 1)):
            inputs, labels, direction = data

            if random.uniform(0, 1) > 0.5:
                inputs = fliplr(inputs)
                direction = 1 - direction

            if inputs.size(0) == 1:
                continue
            inputs = inputs.cuda()
            labels = labels.long().cuda()
            direction = direction.long().cuda()
            b_size = labels.size(0)

            optimizer.zero_grad()

            logits = net(inputs, labels)
            if multi_gpus:
                loss = net.module.get_loss(logits, labels, direction)
            else:
                loss = net.get_loss(logits, labels, direction)

            if loss == 0:
                continue

            acc = accuracy(logits[0].data, labels, topk=(1, 5))
            losses.update(loss.item(), b_size)
            train_acc.update(acc[0], b_size)
            train_acc5.update(acc[1], b_size)

            loss.backward()
            optimizer.step()
        _print('Train Epoch: {}\t'
               'Loss: {loss.avg:.4f}\t'
               'TrainAcc: Prec@1 {train_acc.avg:.3f}%\tPrec@2 {erase_train_acc.avg:.3f}%'.format(
            epoch + 1, loss=losses, train_acc=train_acc, erase_train_acc=train_acc5
        ))

        # val
        if (epoch + 1) % TEST_FREQ == 0:
            net.eval()
            gallery_features = []
            gallery_labels = []
            query_features = []
            query_labels = []
            for data in tqdm(gallery_iter, desc='Train Epoch: {}'.format(epoch + 1)):
                with torch.no_grad():
                    inputs, labels = data
                    if inputs.size(0) == 1:
                        continue
                    labels = labels.long().cuda()
                    b_size = labels.size(0)

                    ff = torch.FloatTensor(b_size, feature_size).zero_().cuda()
                    for i in range(1):
                        flip_inputs = fliplr(inputs).detach()
                        flip_inputs = Variable(flip_inputs.cuda())

                        input_img = Variable(inputs.cuda())
                        if multi_gpus:
                            features = net.module.features(input_img)[0]
                            flip_features = net.module.features(flip_inputs)[0]
                        else:
                            features = net.features(input_img)[0]
                            flip_features = net.features(flip_inputs)[0]

                        ff += torch.cat((features, flip_features), dim=1)

                    fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
                    ff = ff.div(fnorm.expand_as(ff))

                    for i in range(b_size):
                        gallery_features.append(ff[i].cpu().numpy())
                        gallery_labels.append(labels[i].cpu().numpy())

            for data in tqdm(probe_iter, desc='Train Epoch: {}'.format(epoch + 1)):
                with torch.no_grad():
                    inputs, labels = data
                    if inputs.size(0) == 1:
                        continue
                    labels = labels.long().cuda()
                    b_size = labels.size(0)

                    ff = torch.FloatTensor(b_size, feature_size).zero_().cuda()
                    for i in range(1):
                        flip_inputs = fliplr(inputs).detach()
                        flip_inputs = Variable(flip_inputs.cuda())

                        input_img = Variable(inputs.cuda())
                        if multi_gpus:
                            features = net.module.features(input_img)[0]
                            flip_features = net.module.features(flip_inputs)[0]
                        else:
                            features = net.features(input_img)[0]
                            flip_features = net.features(flip_inputs)[0]

                        ff += torch.cat((features, flip_features), dim=1)

                    fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
                    ff = ff.div(fnorm.expand_as(ff))

                    for i in range(b_size):
                        query_features.append(ff[i].cpu().numpy())
                        query_labels.append(labels[i].cpu().numpy())
            gallery_features = torch.FloatTensor(gallery_features)
            gallery_labels = np.array(gallery_labels)
            query_features = torch.FloatTensor(query_features)
            query_labels = np.array(query_labels)

            CMC, ap = evaluate_rerank_CMC(query_features, query_labels, gallery_features, gallery_labels)
            _print('Rank@1:%f Rank@5:%f Rank@10:%f mAP:%f' % (CMC[0], CMC[4], CMC[9], ap / len(query_labels)))
            if max_test_acc <= CMC[0]:
                max_test_acc = CMC[0]
                flag = True
        # save
        if flag:
            msg = 'Saving checkpoint: {}'.format(epoch + 1)
            _print(msg)
            if multi_gpus:
                net_state_dict = net.module.state_dict()
            else:
                net_state_dict = net.state_dict()
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            torch.save(
                {'epoch': epoch,
                 'net_state_dict': net_state_dict},
                os.path.join(save_dir, 'model.ckpt')
            )
    _print('-------max_test_acc Rank@1 {max_test_acc:.3f}-------'.format(
        max_test_acc=max_test_acc
    ))

    _print('finish')


if __name__ == '__main__':
    main()



