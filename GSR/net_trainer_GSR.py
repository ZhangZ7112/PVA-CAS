import os
import shutil
from os.path import join
import numpy as np
from torch.utils.data import DataLoader
import torch
from torch import nn

from parameters_GSR import Parameters
import sys
sys.path.append('../')
from Dataloader_self import DataLoad_update_pseudo
sys.path.append('../../../')
from models.Unet_self import UNet, Proto_UNet
from utils.loss import B_crossentropy, partial_entropy
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

def train_epoch(net, optimizer, criterion, dataloader, epoch, n_epochs, Iters, eta_log):
    loss_log = AverageMeter()
    dice_log = AverageMeter()

    for i in range(Iters):
        net.train()
        net.zero_grad()
        optimizer.zero_grad()

        temp_image, temp_GT, temp_target, image, pseudo, label, refresh_pseudo_path = next(dataloader.__iter__())

        raw_name = refresh_pseudo_path[0].split('/')[-1]
        img_id = raw_name

        if torch.cuda.is_available():
            temp_image = temp_image.cuda()
            temp_target = temp_target.cuda()

        seg = net(temp_image)
        loss = criterion(seg, temp_target)

        loss.backward()
        optimizer.step()

        dice_data = (dice_coef()(temp_target, seg)).data
        dice_log.update(dice_data, temp_target.size(0))
        loss_log.update(loss.data, temp_target.size(0))

        res = '\t'.join(['Epoch: [%d/%d]' % (epoch + 1, n_epochs),
                         'Iter: [%d/%d]' % (i + 1, Iters),
                         'Loss_S %f' % (loss_log.avg),
                         'Dice %f' % (dice_log.avg)])
        print(res)
        if dice_data>=0.5:
            eta_log = refresh_pseudo(net, image[0], pseudo[0], label[0], refresh_pseudo_path[0],eta_log,img_id)

    return eta_log

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

    train_dataset = DataLoad_update_pseudo(p.i_txt, p.img_base, p.GT_lab_base,
                                           p.random_lumen_lab_base, p.pseudo_base,p.update_pseudo_base)
    dataloader = DataLoader(train_dataset, batch_size=p.batch_size, shuffle=True)

    optimizer = torch.optim.Adam(net.parameters(), lr=p.lr)
    loss = B_crossentropy()

    # initialize eta_log
    if os.path.exists(p.eta_log_path):
        eta_log = np.load(p.eta_log_path, allow_pickle=True).item()
    else:
        eta_log = {}

    for epoch in range(p.n_epochs-k):

        eta_log = train_epoch(net, optimizer, loss, dataloader, epoch+k, p.n_epochs, p.Iters, eta_log)
        #saving eta_log
        npy_name = p.eta_log_path
        np.save(npy_name, eta_log)

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

def predict(model, save_path, i_txt, img_base, num_classes, patch_shape, Meanstd_dir, save_float=False):
    print("Predict")
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

        # image = torch.from_numpy(image)
        # print(image.shape)
        # if torch.cuda.is_available():
        #     image = image.cuda()
        # with torch.no_grad():
        #     predict = model(image).data.cpu().numpy()

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

        if save_float:
            predict = predict.astype(dtype=np.float32)
        else:
            predict = np.where(predict[0][0] > 0.5, 1, 0)
            predict = predict.astype(dtype=np.uint16)
        save_name = '{0}/{1}'.format(save_path, image_name)
        predict[:,:,:].tofile(save_name)


def pconv_predict(model, save_path, i_txt, img_base, Meanstd_dir,  patch_shape=(256,512,512)):
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
        image = np.where(image < 0, 0, image)
        image = np.where(image >2048, 2048, image)
        mean, std = np.load(Meanstd_dir)
        image = (image - mean) / std

        shape = patch_shape

        z, y, x = image.shape[0], image.shape[1], image.shape[2]

        predict = np.zeros([ z, y, x], dtype=np.float32)
        n_map = np.zeros([z, y, x], dtype=np.float32)

        a = np.zeros(shape=shape)
        a = np.where(a == 0)
        map_kernal = 1 / ((a[0] - shape[0] // 2) ** 4 + (a[1] - shape[1] // 2) ** 4 + (a[2] - shape[2] // 2) ** 4 + 1)
        map_kernal = np.reshape(map_kernal, shape)

        image = image[np.newaxis, np.newaxis, :, :, :]
        stride_z = shape[0] // 2

        for i in range(z // stride_z - 1):
            image_i = image[:, :, i * stride_z:i * stride_z + shape[0], :, :]
            image_i = torch.from_numpy(image_i)
            if torch.cuda.is_available():
                image_i = image_i.cuda()
            output = model(image_i,start_pconv=True)
            output = output.data.cpu().numpy()[0,0]    #[256,512,512]

            predict[i * stride_z:i * stride_z + shape[0], :, :] += output * map_kernal
            n_map[i * stride_z:i * stride_z + shape[0], :, :] += map_kernal

        image_i = image[:, :, z - shape[0]:z, :, :]
        image_i = torch.from_numpy(image_i)
        if torch.cuda.is_available():
            image_i = image_i.cuda()
        output = model(image_i, start_pconv=True)
        output = output.data.cpu().numpy()[0, 0]  #[256,512,512]

        predict[z - shape[0]:z, :, :] += output * map_kernal
        n_map[z - shape[0]:z, :, :] += map_kernal

        predict = predict / n_map
        predict = predict.astype(dtype=np.float32)

        save_name = '{0}/{1}'.format(save_path, image_name)
        predict.tofile(save_name)

def refresh_pseudo(model, image, pseudo, label, new_pseudo_path,eta_log,img_id,num_classes=1, eta_threshold=0.65, patch_shape=(256,512,512)):
    print('refresh_pseudo')
    model.eval()

    shape = patch_shape
    z, y, x = image.shape[0], image.shape[1], image.shape[2]

    predict = np.zeros([1, num_classes, z, y, x], dtype=np.float32)
    n_map = np.zeros([1, num_classes, z, y, x], dtype=np.float32)

    a = np.zeros(shape=shape)
    a = np.where(a == 0)
    map_kernal = 1 / ((a[0] - shape[0] // 2) ** 4 + (a[1] - shape[1] // 2) ** 4 + (a[2] - shape[2] // 2) ** 4 + 1)
    map_kernal = np.reshape(map_kernal, newshape=(1, 1,) + shape)

    image = image[np.newaxis, np.newaxis, :, :, :]
    stride_z = shape[0] // 2


    for i in range(z // stride_z - 1):
        image_i = image[:,:, i * stride_z:i * stride_z + shape[0], :,:]
        if torch.cuda.is_available():
            image_i = image_i.cuda()
        output = model(image_i)
        output = output.data.cpu().numpy()

        predict[:,:, i * stride_z:i * stride_z + shape[0], :,:] += output * map_kernal
        n_map[:,:, i * stride_z:i * stride_z + shape[0], :,:] += map_kernal

    image_i = image[:, :, z - shape[0]:z, :,:]
    if torch.cuda.is_available():
        image_i = image_i.cuda()
    output = model(image_i)
    output = output.data.cpu().numpy()

    predict[:, :, z - shape[0]:z, :,:] += output * map_kernal
    n_map[:, :, z - shape[0]:z, :,:] += map_kernal

    predict = predict / n_map
    predict = predict[0][0].astype(dtype=np.float32)

    eta = get_eta(predict, label.data.cpu().numpy())
    print('eta:',eta)
    if img_id not in eta_log:
        eta_log[img_id] = 0.0
    if (eta>=eta_threshold) and ((eta-eta_log[img_id])>=0.05):
        new_pseudo_name = get_new_pseudo(eta, predict, pseudo.data.cpu().numpy(), new_pseudo_path)
        print('refreshing pseudo of {0} with eta={1}; pre eta_log[{2}]={3}'.format(new_pseudo_name, eta,img_id, eta_log[img_id]))
        eta_log[img_id] = eta

    return eta_log

def get_eta(predict, label):
    #calculating the confidecnce level--eta
    smooth = 1e-6
    num = np.sum(label)

    return np.sum(predict*label)/(num+smooth)

def get_new_pseudo(eta, predict, pseudo, new_pseudo_path):
    new_pseudo = (1-(eta))*pseudo + (eta)*predict
    new_pseudo = new_pseudo.astype(dtype=np.float32)
    new_pseudo.tofile(new_pseudo_path)
    return new_pseudo_path.split('/')[-1]

def Dice(pre_txt, label_base, pred_save_path):
    file = read_file_from_txt(pre_txt)
    file_num = len(file)

    dice = np.zeros(shape=(file_num), dtype=np.float32)
    for i in range(file_num):
        s = file[i]

        predict = np.fromfile(join(pred_save_path,s), dtype=np.uint16)
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

def apply_dropout(m):
    if type(m) == nn.Dropout:
        m.train()


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
    net = Proto_UNet(in_channels=1, base_channels=16)
    train_net(net, p, k=0)

    if torch.cuda.is_available():
        if torch.cuda.device_count()>1:
            net = nn.DataParallel(net).cuda()
        else:
            net = net.cuda()

    checkpoint = '{0}/{1}_best_epoch.pth'.format(p.checkpoint_dir, p.model_name)
    net.load_state_dict(torch.load(checkpoint), False)
    print('loaded:',checkpoint)

    predict(net, save_path=p.t_save_path, i_txt=p.t_txt, img_base=p.img_base, num_classes=1,patch_shape=p.patch_shape, Meanstd_dir=p.Meanstd_dir)
    Dice(pre_txt=p.t_txt, label_base=p.GT_lab_base, pred_save_path=p.t_save_path)


