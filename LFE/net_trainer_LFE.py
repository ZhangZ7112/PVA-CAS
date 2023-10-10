import os
import shutil
from os.path import join
import numpy as np
from torch.utils.data import DataLoader
import torch
from torch import nn

import sys
from parameters_LFE import Parameters
sys.path.append('../')
from Dataloader_patch import DataLoad_patch
sys.path.append('../../../')
from models.Unet_self import UNet
from utils.loss import B_crossentropy
from utils.loss import dice_coef



class AverageMeter(object):
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def train_epoch(net, optimizer, criterion, dataloader, epoch, n_epochs, Iters):
    loss_log = AverageMeter()
    dice_log = AverageMeter()

    net.train()
    for i in range(Iters):
        net.zero_grad()
        optimizer.zero_grad()

        input, target = next(dataloader.__iter__())
        if torch.cuda.is_available():
            input = input.cuda()
            target = target.cuda()

        seg = net(input)
        loss = criterion(seg, target)

        loss.backward()
        optimizer.step()

        dice_log.update((dice_coef()(target, seg)).data, target.size(0))
        loss_log.update(loss.data, target.size(0))

        res = '\t'.join(['Epoch: [%d/%d]' % (epoch + 1, n_epochs),
                         'Iter: [%d/%d]' % (i + 1, Iters),
                         'Loss_S %f' % (loss_log.avg),
                         'Dice %f' % (dice_log.avg)])
        print(res)

    return

def train_net(net, p, k=0):
    if k!=0:
        checkpoint = '{0}/{1}_epoch_{2}.pth'.format(p.checkpoint_dir, p.model_name, k)
        net.load_state_dict(torch.load(checkpoint), False)
        print('loaded:',checkpoint)

    dice_max = 0
    epoch_max = 0

    if torch.cuda.is_available():
        if torch.cuda.device_count()>1:
            net = nn.DataParallel(net).cuda()
        else:
            net = net.cuda()

    train_dataset = DataLoad_patch(p.i_txt, p.img_base, p.train_lab_base, p.patch_shape)
    dataloader = DataLoader(train_dataset, batch_size=p.batch_size, shuffle=True)

    optimizer = torch.optim.Adam(net.parameters(), lr=p.lr)
    loss = B_crossentropy()

    for epoch in range(p.n_epochs-k):
        train_epoch(net, optimizer, loss, dataloader, epoch+k, p.n_epochs, p.Iters)
        if ((k + epoch + 1) % 10) == 0:
            torch.save(net.state_dict(), '{0}/{1}_epoch_{2}.pth'.format(p.checkpoint_dir, p.model_name, k + epoch + 1))

        if ((k + epoch + 1) >= 175):
            predict(net, save_path=p.v_save_path, i_txt=p.v_txt, img_base=p.img_base, num_classes=1, patch_shape=p.patch_shape, Meanstd_dir=p.Meanstd_dir)
            epoch_dice = Dice(p.v_txt,p.GT_lab_base, p.v_save_path)
            mean_dice = np.mean(epoch_dice)
            if (mean_dice > dice_max):
                dice_max = mean_dice
                epoch_max = k + epoch + 1
                all_list = os.listdir(p.v_save_path)
                for thing in all_list:
                    old_name = join(p.v_save_path,thing)
                    new_name = join(p.best_test_path,thing)
                    shutil.copyfile(old_name, new_name)

                torch.save(net.state_dict(), '{0}/{1}_best_epoch.pth'.format(p.checkpoint_dir, p.model_name))
                print('best_epoch:{0} ; Dice_on_validation:{1}'.format(epoch_max, dice_max))

    print('best_epoch:{0} ; Dice_on_validation:{1}'.format(epoch_max, dice_max))
    torch.cuda.empty_cache()

def predict(model, save_path, i_txt, img_base, num_classes, patch_shape, Meanstd_dir, binary_output=True):
    print("Predict test data")
    model.eval()

    image_file = read_file_from_txt(i_txt)
    for image_name in image_file:
        name = img_base + image_name
        print(name)
        image = np.fromfile(file=name, dtype=np.int16)
        z,x,y = int(image.shape[0]/(512*512)), 512, 512
        raw_shape = (z, 512, 512)
        image = image.reshape(raw_shape)
        image = image.astype(np.float32)

        #normalization
        image = np.where(image < 0, 0, image)
        image = np.where(image >2048, 2048, image)
        mean, std = np.load(Meanstd_dir)
        image = (image - mean) / std

        shape = patch_shape

        predict = np.zeros([1, num_classes, z, y, x], dtype=np.float32)
        n_map = np.zeros([1, num_classes, z, y, x], dtype=np.float32)

        a = np.zeros(shape=shape)
        a = np.where(a == 0)
        map_kernal = 1 / ((a[0] - shape[0] // 2) ** 4 + (a[1] - shape[1] // 2) ** 4 + (a[2] - shape[2] // 2) ** 4 + 1)
        map_kernal = np.reshape(map_kernal, newshape=(1, 1,) + shape)

        image = image[np.newaxis,np.newaxis, :, :, :]
        stride_x = shape[0] // 2
        stride_y = shape[1] // 2
        stride_z = shape[2] // 2

        for i in range(z // stride_x - 1):
            for j in range(y // stride_y - 1):
                for k in range(x // stride_z - 1):
                    image_i = image[:, :, i * stride_x:i * stride_x + shape[0], j * stride_y:j * stride_y + shape[1],
                              k * stride_z:k * stride_z + shape[2]]
                    image_i = torch.from_numpy(image_i)
                    if torch.cuda.is_available():
                        image_i = image_i.cuda()
                    output = model(image_i)
                    output = output.data.cpu().numpy()

                    predict[:, :, i * stride_x:i * stride_x + shape[0], j * stride_y:j * stride_y + shape[1],
                    k * stride_z:k * stride_z + shape[2]] += output * map_kernal

                    n_map[:, :, i * stride_x:i * stride_x + shape[0], j * stride_y:j * stride_y + shape[1],
                    k * stride_z:k * stride_z + shape[2]] += map_kernal

                image_i = image[:, :, i * stride_x:i * stride_x + shape[0], j * stride_y:j * stride_y + shape[1],
                          x - shape[2]:x]
                image_i = torch.from_numpy(image_i)
                if torch.cuda.is_available():
                    image_i = image_i.cuda()
                output = model(image_i)
                output = output.data.cpu().numpy()
                predict[:, :, i * stride_x:i * stride_x + shape[0], j * stride_y:j * stride_y + shape[1],
                x - shape[2]:x] += output * map_kernal

                n_map[:, :, i * stride_x:i * stride_x + shape[0], j * stride_y:j * stride_y + shape[1],
                x - shape[2]:x] += map_kernal

            for k in range(x // stride_z - 1):
                image_i = image[:, :, i * stride_x:i * stride_x + shape[0], y - shape[1]:y,
                          k * stride_z:k * stride_z + shape[2]]
                image_i = torch.from_numpy(image_i)
                if torch.cuda.is_available():
                    image_i = image_i.cuda()
                output = model(image_i)
                output = output.data.cpu().numpy()
                predict[:, :, i * stride_x:i * stride_x + shape[0], y - shape[1]:y,
                k * stride_z:k * stride_z + shape[2]] += output * map_kernal

                n_map[:, :, i * stride_x:i * stride_x + shape[0], y - shape[1]:y,
                k * stride_z:k * stride_z + shape[2]] += map_kernal

            image_i = image[:, :, i * stride_x:i * stride_x + shape[0], y - shape[1]:y, x - shape[2]:x]
            image_i = torch.from_numpy(image_i)
            if torch.cuda.is_available():
                image_i = image_i.cuda()
            output = model(image_i)
            output = output.data.cpu().numpy()

            predict[:, :, i * stride_x:i * stride_x + shape[0], y - shape[1]:y, x - shape[2]:x] += output * map_kernal
            n_map[:, :, i * stride_x:i * stride_x + shape[0], y - shape[1]:y, x - shape[2]:x] += map_kernal

        for j in range(y // stride_y - 1):
            for k in range((x - shape[2]) // stride_z):
                image_i = image[:, :, z - shape[0]:z, j * stride_y:j * stride_y + shape[1],
                          k * stride_z:k * stride_z + shape[2]]
                image_i = torch.from_numpy(image_i)
                if torch.cuda.is_available():
                    image_i = image_i.cuda()
                output = model(image_i)
                output = output.data.cpu().numpy()

                predict[:, :, z - shape[0]:z, j * stride_y:j * stride_y + shape[1],
                k * stride_z:k * stride_z + shape[2]] += output * map_kernal

                n_map[:, :, z - shape[0]:z, j * stride_y:j * stride_y + shape[1],
                k * stride_z:k * stride_z + shape[2]] += map_kernal

            image_i = image[:, :, z - shape[0]:z, j * stride_y:j * stride_y + shape[1],
                      x - shape[2]:x]
            image_i = torch.from_numpy(image_i)
            if torch.cuda.is_available():
                image_i = image_i.cuda()
            output = model(image_i)
            output = output.data.cpu().numpy()

            predict[:, :, z - shape[0]:z, j * stride_y:j * stride_y + shape[1],
            x - shape[2]:x] += output * map_kernal

            n_map[:, :, z - shape[0]:z, j * stride_y:j * stride_y + shape[1],
            x - shape[2]:x] += map_kernal

        for k in range(x // stride_z - 1):
            image_i = image[:, :, z - shape[0]:z, y - shape[1]:y,
                      k * stride_z:k * stride_z + shape[2]]
            image_i = torch.from_numpy(image_i)
            if torch.cuda.is_available():
                image_i = image_i.cuda()
            output = model(image_i)
            output = output.data.cpu().numpy()

            predict[:, :, z - shape[0]:z, y - shape[1]:y,
            k * stride_z:k * stride_z + shape[2]] += output * map_kernal

            n_map[:, :, z - shape[0]:z, y - shape[1]:y,
            k * stride_z:k * stride_z + shape[2]] += map_kernal

        image_i = image[:, :, z - shape[0]:z, y - shape[1]:y, x - shape[2]:x]
        image_i = torch.from_numpy(image_i)
        if torch.cuda.is_available():
            image_i = image_i.cuda()
        output = model(image_i)
        output = output.data.cpu().numpy()

        predict[:, :, z - shape[0]:z, y - shape[1]:y, x - shape[2]:x] += output * map_kernal
        n_map[:, :, z - shape[0]:z, y - shape[1]:y, x - shape[2]:x] += map_kernal

        predict = predict / n_map

        if binary_output == True:
            predict = np.where(predict[0][0] > 0.5, 1, 0)
            predict = predict.astype(dtype=np.uint16)
        else:
            predict = predict[0, 0]
            predict = predict.astype(dtype=np.float32)

        save_name = '{0}/{1}'.format(save_path, image_name)
        predict[:,:,:].tofile(save_name)


def Dice(pre_txt, label_base, pred_save_path, type=np.uint16):
    file = read_file_from_txt(pre_txt)
    file_num = len(file)

    dice = np.zeros(shape=(file_num), dtype=np.float32)
    for i in range(file_num):
        s = file[i]

        predict = np.fromfile(join(pred_save_path,s), dtype=type)
        predict = np.where(predict > 0.5, 1, 0)

        groundtruth = np.fromfile(label_base+s, dtype=np.uint16)
        groundtruth = np.where(groundtruth > 0, 1, 0)

        tmp = predict + groundtruth
        a = np.sum(np.where(tmp == 2, 1, 0))
        b = np.sum(predict)
        c = np.sum(groundtruth)

        dice[i] = (2 * a) / (b + c)

        print(s, dice[i])

    print('Dice_AVG=', np.mean(dice))

    return dice


def read_file_from_txt(txt_path):
    files=[]
    for line in open(txt_path, 'r'):
        files.append(line.strip())
    return files

def reshape_img(image, z, y, x):
    out = np.zeros([z, y, x], dtype=np.float32)
    out[0:image.shape[0], 0:image.shape[1], 0:image.shape[2]] = image[0:image.shape[0], 0:image.shape[1],
                                                                0:image.shape[2]]
    return out

if __name__ == '__main__':
    p = Parameters()
    p.show_p()
    if not os.path.exists(p.checkpoint_dir):
        os.makedirs(p.checkpoint_dir)
    if not os.path.exists(p.v_save_path):
        os.makedirs(p.v_save_path)
    if not os.path.exists(p.t_save_path):
        os.makedirs(p.t_save_path)
    if not os.path.exists(p.best_test_path):
        os.makedirs(p.best_test_path)

    os.environ["CUDA_VISIBLE_DEVICES"] = p.CUDA_VISIBLE_DEVICES
    net = UNet(in_channels=1, base_channels=16)
    train_net(net, p, k=0)
    if torch.cuda.is_available():
        if torch.cuda.device_count()>1:
            net = nn.DataParallel(net).cuda()
        else:
            net = net.cuda()

    checkpoint = '{0}/{1}_best_epoch.pth'.format(p.checkpoint_dir, p.model_name)
    net.load_state_dict(torch.load(checkpoint), False)
    print('loaded:',checkpoint)

    predict(net, save_path=p.t_save_path, i_txt=p.t_txt, img_base=p.img_base,num_classes=1, patch_shape=p.patch_shape, Meanstd_dir=p.Meanstd_dir)
    Dice(pre_txt=p.t_txt, label_base=p.GT_lab_base, pred_save_path=p.t_save_path)


