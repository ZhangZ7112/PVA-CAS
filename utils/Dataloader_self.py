from torch.utils import data
import numpy as np
import os

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".raw"])

def read_file_from_txt(txt_path):
    files=[]
    for line in open(txt_path, 'r'):
        files.append(line.strip())

    return files

def to_categorical(y, num_classes=None):
    y = np.array(y, dtype='int')
    input_shape = y.shape

    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((num_classes, n))
    categorical[y, np.arange(n)] = 1

    output_shape = (num_classes,) + input_shape
    categorical = np.reshape(categorical, output_shape)

    return categorical

def reshape_img(image, x, y, z):
    out = np.zeros([x, y, z], dtype=np.float32)
    for i in range(image.shape[0]):
        out[i] = image[i]

    return out


class DataLoad_patch(data.Dataset):
    def __init__(self, i_txt, img_base,lab_base, patch_shape=(128,128,128)):
        super(DataLoad_patch, self).__init__()
        # self.image_filenames = [x for x in listdir(join(image_dir)) if is_image_file(x)]
        self.image_file = read_file_from_txt(i_txt)
        self.label_file = read_file_from_txt(i_txt)
        self.img_base = img_base
        self.lab_base = lab_base
        self.patch_shape = patch_shape
        self.mean, self.std = 840.0, 447.6             # statistical parameters of the dataset

    def __getitem__(self, index):
        image_path = self.img_base + self.image_file[index]
        label_path = self.lab_base + self.label_file[index]
        image = np.fromfile(file=image_path, dtype=np.int16)
        target = np.fromfile(file=label_path, dtype=np.uint16)

        z = int(image.shape[0]/(512*512))
        shape = (z, 512, 512)

        image = image.reshape(shape)
        target = target.reshape(shape)
        image = image.astype(np.float32)
        target = target.astype(np.float32)

        image = np.where(image < 0, 0.0, image)
        image = np.where(image > 2048, 2048, image)
        target = np.where(target > 0, 1, target)
        image = (image - self.mean) / self.std
        #target = to_categorical(target, 2)

        while True:
            center_z = np.random.randint(0, shape[0] - self.patch_shape[0], 1, dtype=np.int)[0]
            if (shape[1] - self.patch_shape[1]) > 0:
                center_y = np.random.randint(0, shape[1] - self.patch_shape[1], 1, dtype=np.int)[0]
            else:
                center_y = 0
            if (shape[2] - self.patch_shape[2])>0:
                center_x = np.random.randint(0, shape[2] - self.patch_shape[2], 1, dtype=np.int)[0]
            else:
                center_x = 0

            temp_image = image[center_z:self.patch_shape[0] +
                                   center_z, center_y:self.patch_shape[1] + center_y, center_x:self.patch_shape[2] + center_x]
            temp_target = target[center_z:self.patch_shape[0] +
                                     center_z, center_y:self.patch_shape[1] + center_y, center_x:self.patch_shape[2] + center_x]

            if np.max(temp_target)==1:
                break

        image = temp_image[np.newaxis, :, :, :]  # b c d w h
        target = temp_target[np.newaxis, :, :, :]

        return image, target

    def __len__(self):
        return len(self.image_file)


class DataLoad_update_pseudo(data.Dataset):
    def __init__(self, i_txt, img_base, GT_base, random_lumen_base, pseudo_base,update_pseudo_base,patch_shape=(256,512,512)):
        super(DataLoad_update_pseudo, self).__init__()
        self.image_file = read_file_from_txt(i_txt)
        self.label_file = read_file_from_txt(i_txt)
        self.img_base = img_base
        self.GT_base = GT_base
        self.random_lumen_base = random_lumen_base
        self.pseudo_base = pseudo_base
        self.update_pseudo_base = update_pseudo_base
        self.patch_shape = patch_shape
        self.mean, self.std = 840.0, 447.6                        # statistical parameters of the dataset

    def __getitem__(self, index):
        img_name = self.image_file[index]
        image_path = self.img_base + img_name
        GT_path = self.GT_base + img_name
        pseudo_path = self.pseudo_base + img_name
        update_pseudo_path = self.update_pseudo_base + img_name
        label_path = self.random_lumen_base + img_name

        image = np.fromfile(file=image_path, dtype=np.int16)
        GT = np.fromfile(file=GT_path, dtype=np.uint16)
        if os.path.exists(update_pseudo_path):
            pseudo = np.fromfile(file=update_pseudo_path, dtype=np.float32)
        else:
            pseudo = np.fromfile(file=pseudo_path, dtype=np.float32)
        label = np.fromfile(file=label_path, dtype=np.uint16)

        z = int(image.shape[0]/(512*512))
        shape = (z, 512, 512)

        image = image.reshape(shape)
        GT = GT.reshape(shape)
        pseudo = pseudo.reshape(shape)
        label = label.reshape(shape)

        target = pseudo + label

        image = image.astype(np.float32)
        GT = GT.astype(np.float32)
        pseudo = pseudo.astype(np.float32)
        label = label.astype(np.float32)
        target = target.astype(np.float32)

        image = np.where(image < 0, 0.0, image)
        image = np.where(image > 2048, 2048, image)
        target = np.where(target > 1.0, 1.0, target)
        image = (image - self.mean) / self.std

        while True:
            center_z = np.random.randint(0, shape[0] - self.patch_shape[0], 1, dtype=np.int)[0]
            if (shape[1] - self.patch_shape[1]) > 0:
                center_y = np.random.randint(0, shape[1] - self.patch_shape[1], 1, dtype=np.int)[0]
            else:
                center_y = 0
            if (shape[2] - self.patch_shape[2])>0:
                center_x = np.random.randint(0, shape[2] - self.patch_shape[2], 1, dtype=np.int)[0]
            else:
                center_x = 0

            temp_image = image[center_z:self.patch_shape[0] +
                                   center_z, center_y:self.patch_shape[1] + center_y, center_x:self.patch_shape[2] + center_x]
            temp_target = target[center_z:self.patch_shape[0] +
                                     center_z, center_y:self.patch_shape[1] + center_y, center_x:self.patch_shape[2] + center_x]
            temp_GT = GT[center_z:self.patch_shape[0] +
                                     center_z, center_y:self.patch_shape[1] + center_y, center_x:self.patch_shape[2] + center_x]
            if np.max(temp_target)>=0.8:
                break

        temp_image = temp_image[np.newaxis, :, :, :]  # b c d w h
        temp_GT = temp_GT[np.newaxis, :, :, :]
        temp_target = temp_target[np.newaxis, :, :, :]

        return temp_image, temp_GT, temp_target, image, pseudo, label, update_pseudo_path

    def __len__(self):
        return len(self.image_file)