import torch
import os, random
import cv2
import numpy, math
from torch.utils.data import Dataset, DataLoader
from .bool import get_bool
from.logs import log

class SR_dataset_RGB(Dataset):
    def __init__(self, HR_folder, LR_folder, scale, patch_size, train=True, is_post=True, normal=False, is_real=False, test_patch=None):
        self.scale = scale
        self.HR_folder = HR_folder
        self.LR_folder = LR_folder
        self.is_real = is_real
        if self.is_real:
            self.HR_folder += (str(self.scale) + '/')
            self.LR_folder += (str(self.scale) + '/')
        else:
            self.LR_folder += (str(self.scale) + '/')
        self.hr_img = os.listdir(self.HR_folder)
        self.lr_img = os.listdir(self.LR_folder)
        self.patch_size = patch_size
        self.train = train
        self.is_post = is_post
        self.normal = normal
        self.cut = test_patch
    def __len__(self):
        return len(self.hr_img)
    def get_patch(self, lr, hr):
        h, w, _ = lr.shape
        randh = random.randrange(0, h - self.patch_size)
        randw = random.randrange(0, w - self.patch_size)
        toh = randh + self.patch_size
        tow = randw + self.patch_size
        lr_patch = lr[randh:toh, randw:tow ,:]
        if self.is_post:
            hr_patch = hr[randh*self.scale : toh*self.scale, randw*self.scale : tow*self.scale, :]
        else:
            hr_patch = hr[randh : toh, randw : tow, :]
        return lr_patch,  hr_patch
    def cut_pic(self, lr):
        [c, h_lr, w_lr] = lr.shape
        h_n = math.ceil(h_lr / self.cut)
        w_n = math.ceil(w_lr / self.cut)
        patch_num = h_n * w_n
        lr_patch = torch.zeros([patch_num, c, self.cut, self.cut])
        n = 0
        for i in range(h_n):
            for j in range(w_n):
                if ((i + 1) == h_n) and ((j + 1) == w_n):
                    lr_patch[n, :, :, :] = lr[ :, h_lr-self.cut:, w_lr-self.cut:]
                elif (i + 1) == h_n:
                    lr_patch[n, :, :, :] = lr[:, h_lr-self.cut:, j*self.cut:(j+1)*self.cut]
                elif (j + 1) == w_n:
                    lr_patch[n, :, :, :] = lr[:, i*self.cut:(i+1)*self.cut, w_lr-self.cut:]
                else:
                    lr_patch[n, :, :, :] = lr[:, i*self.cut:(i+1)*self.cut, j*self.cut:(j+1)*self.cut]
                n += 1
        return lr_patch, [h_lr, w_lr]
    def __getitem__(self, index):
        hr_name = self.hr_img[index]
        lr_name = self.lr_img[index]
        hr_path = os.path.join(self.HR_folder, hr_name)
        lr_path = os.path.join(self.LR_folder, lr_name)
        hr = torch.from_numpy(cv2.imread(hr_path))
        lr = torch.from_numpy(cv2.imread(lr_path))
        if self.train:
            lr, hr = self.get_patch(lr, hr)
        hr = hr.permute(2,0,1).float()
        lr = lr.permute(2,0,1).float()
        if self.normal:
            hr = hr / 255.
            lr = lr / 255.
        if not self.cut == None:
            lr, shape = self.cut_pic(lr)
            return lr, hr, shape
        return lr, hr

class SR_dataset_Y(SR_dataset_RGB):
    def get_patch(self, lr, hr):
        h, w = lr.shape
        randh = random.randrange(0, h - self.patch_size)
        randw = random.randrange(0, w - self.patch_size)
        toh = randh + self.patch_size
        tow = randw + self.patch_size
        lr_patch = lr[randh : toh, randw : tow]
        if self.is_post:
            hr_patch = hr[randh*self.scale : toh*self.scale, randw*self.scale : tow*self.scale]
        else:
            hr_patch = hr[randh : toh, randw : tow]
        return lr_patch,  hr_patch
    def cut_pic(self, lr):
        [c, h_lr, w_lr] = lr.shape
        h_n = math.ceil(h_lr / self.cut)
        w_n = math.ceil(w_lr / self.cut)
        patch_num = h_n * w_n
        lr_patch = torch.zeros([patch_num, c, self.cut, self.cut])
        n = 0
        for i in range(h_n):
            for j in range(w_n):
                if ((i + 1) == h_n) and ((j + 1) == w_n):
                    lr_patch[n, :, :, :] = lr[ :, h_lr-self.cut:, w_lr-self.cut:]
                elif (i + 1) == h_n:
                    lr_patch[n, :, :, :] = lr[:, h_lr-self.cut:, j*self.cut:(j+1)*self.cut]
                elif (j + 1) == w_n:
                    lr_patch[n, :, :, :] = lr[:, i*self.cut:(i+1)*self.cut, w_lr-self.cut:]
                else:
                    lr_patch[n, :, :, :] = lr[:, i*self.cut:(i+1)*self.cut, j*self.cut:(j+1)*self.cut]
                n += 1
        return lr_patch, [h_lr, w_lr]
    def __getitem__(self, index):
        hr_name = self.hr_img[index]
        lr_name = self.lr_img[index]
        hr_path = os.path.join(self.HR_folder, hr_name)
        lr_path = os.path.join(self.LR_folder, lr_name)
        hr = torch.from_numpy(cv2.imread(hr_path, cv2.IMREAD_GRAYSCALE))
        lr = torch.from_numpy(cv2.imread(lr_path, cv2.IMREAD_GRAYSCALE))
        if self.train:
            lr, hr = self.get_patch(lr, hr)
        hr = hr.unsqueeze(0).float()
        lr = lr.unsqueeze(0).float()
        if self.normal:
            hr = hr / 255.
            lr = lr / 255.
        if not self.cut == None:
            lr, shape = self.cut_pic(lr)
            return lr, hr, shape
        return lr, hr

class SR_demo(Dataset):
    def __init__(self, folder, scale, normal=False, is_input=False, is_Y=False):
        self.folder = folder
        self.scale = scale
        self.normal = normal
        self.is_input = is_input
        self.is_Y = is_Y
        if not self.is_input:
            if is_Y:
                self.folder += '_Y'
            self.folder += '_LR/'
            self.folder += str(self.scale)
        self.folder += '/'
        self.img = os.listdir(self.folder)
    def get_Y(self, img):
        m = numpy.array([65.481, 128.553, 24.966], dtype='float32')
        shape = img.shape
        if len(shape) == 3:
            img = img.reshape((shape[0] * shape[1], 3))
            shape = shape[0 : 2]
        y = numpy.dot(img, m.transpose() / 255.)
        y += 16.
        y = y.reshape(shape)
        return y
    def __len__(self):
        return len(self.img)
    def __getitem__(self, index):
        name = self.img[index]
        path = os.path.join(self.folder, name)
        if self.is_Y and (not self.is_input):
            im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        else:
            im = cv2.imread(path)
        if self.is_input and self.is_Y:
            im = self.get_Y(im)
        im = torch.from_numpy(im)
        if self.is_Y:
            im = im.unsqueeze(0).float()
        else:
            im = im.permute(2,0,1).float()
        if self.normal:
            im = im / 255.
        return im, name

def get_demo_loader(folder, scale, normal, is_input, is_Y):
    data = SR_demo(folder, scale, normal, is_input, is_Y)
    return DataLoader(data, 1, False, num_workers=0, drop_last=False, pin_memory=False)

class Data():
    def __init__(self, sys_conf, data_config, train=True, test_patch=None):
        self.train = train
        self.model_name = sys_conf.model_name
        self.dataset = sys_conf.dataset
        self.batch_size = (sys_conf.batch_size if train else 1)
        self.scale = 1
        self.normal = get_bool(data_config["normalize"])
        self.shuffle = (get_bool(data_config["shuffle"]) if self.train else False)
        self.pic_pair = False
        self.test_patch = test_patch
        if sys_conf.color_channel == "RGB":
            self.color_is_RGB = True
        elif sys_conf.color_channel == "Y":
            self.color_is_RGB = False
        if sys_conf.model_mode == "pre":
            self.is_post = False
        elif sys_conf.model_mode == "post":
            self.is_post = True
        self.patch_size = sys_conf.patch_size
        self.num_workers = (data_config["num_workers"] if self.train else 0)
        self.drop_last = False
        if "drop_last" in data_config:
            self.drop_last = (get_bool(data_config["drop_last"])if self.train else False)
        self.pin_memory = True
        if "pic_pair" in data_config:
            self.pic_pair = get_bool(data_config["pic_pair"])
        if "pin_memory" in data_config:
            self.pin_memory = get_bool(data_config["pin_memory"])
    def show(self):
        log("--------This is dataset and dataloader config--------", self.model_name)
        log("Dataloader num workers: " + str(self.num_workers), self.model_name)
        log("Dataset is pair: " + str(self.pic_pair), self.model_name)
        log("Shuffle: " + str(self.shuffle), self.model_name)
        log("Drop the last batch: " + str(self.drop_last), self.model_name)
        log("Using pin menory: " + str(self.pin_memory), self.model_name)
        log("Using normalization: " + str(self.normal), self.model_name)
    def __set_dataset_path(self):
        if not self.pic_pair:
            HR_folder = './data/'
            if self.train:
                HR_folder += 'train/'
            else:
                HR_folder += 'test/'
            HR_folder += self.dataset
            if not self.color_is_RGB:
                HR_folder += '_Y'
            LR_folder = HR_folder + '_LR/'
            HR_folder += '/'
        else:
            HR_folder = './data/'
            LR_folder = './data/'
            if self.train:
                HR_folder += 'train/'
                LR_folder += 'train/'
            else:
                HR_folder += 'test/'
                LR_folder += 'test/'
            HR_folder += self.dataset
            LR_folder += self.dataset
            HR_folder += '/HR'
            LR_folder += '/LR'
            if not self.color_is_RGB:
                HR_folder += '_Y'
                LR_folder += '_Y'
            HR_folder += '/'
            LR_folder += '/'
        return LR_folder, HR_folder
    def update_scale(self, scale):
        self.scale = scale
    def get_loader(self):
        LR_folder, HR_folder = self.__set_dataset_path()
        if self.color_is_RGB:
            data = SR_dataset_RGB(HR_folder, LR_folder, self.scale, self.patch_size, self.train, self.is_post, self.normal, self.pic_pair, self.test_patch)
        else:
            data = SR_dataset_Y(HR_folder, LR_folder, self.scale, self.patch_size, self.train, self.is_post, self.normal, self.pic_pair, self.test_patch)
        loader = DataLoader(data, self.batch_size, self.shuffle, num_workers=self.num_workers, 
                            drop_last=self.drop_last, pin_memory=self.pin_memory)
        return loader
    def update_dataset(self, dataset):
        self.dataset = dataset