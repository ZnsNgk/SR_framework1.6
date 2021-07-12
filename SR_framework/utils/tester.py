import torch
import os
import utils
import numpy
import models

class Tester():
    def __init__(self, sys_config, data_config):
        self.sys_conf = sys_config
        self.data_config = data_config
        self.is_normal = False
        self.curr_dataset = self.sys_conf.test_dataset[0]
        self.curr_scale = 1
        self.is_pkl = False
        self.test_path = './trained_model/' + self.sys_conf.model_name + '/'
        self.save_path = './test_result/' + self.sys_conf.model_name + '/'
        if self.sys_conf.test_file != "":
            self.test_path += self.sys_conf.test_file
            if 'pkl' in self.sys_conf.test_file:
                self.is_pkl = True
                self.curr_scale = int(self.sys_conf.test_file.replace(".pkl", "").replace("x", ""))
                self.sys_conf.scale_factor = [self.curr_scale]
            else:
                state_list = self.sys_conf.test_file.split('_')
                self.curr_scale = int(state_list[1].replace('x', ''))
                self.sys_conf.scale_factor = [self.curr_scale]
        self.__check_result_folder()
    def __check_result_folder(self):
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
    def __get_model(self):
        if self.sys_conf.model_mode == "post":
            if self.sys_conf.scale_pos == "init":
                if self.sys_conf.model_args == None:
                    return models.get_model(self.sys_conf.model, scale=self.curr_scale)
                else:
                    return models.get_model(self.sys_conf.model, scale=self.curr_scale, **self.sys_conf.model_args)
            elif self.sys_conf.scale_pos == "forward":
                if self.sys_conf.model_args == None:
                    return models.get_model(self.sys_conf.model)
                else:
                    return models.get_model(self.sys_conf.model, **self.sys_conf.model_args)
        elif self.sys_conf.model_mode == "pre":
            if self.sys_conf.model_args == None:
                return models.get_model(self.sys_conf.model)
            else:
                return models.get_model(self.sys_conf.model, **self.sys_conf.model_args)
        else:
            raise NameError("ERROR model_mode!")
    def __test_once(self):
        if self.sys_conf.shave_is_scale:
            self.sys_conf.shave = self.curr_scale
        self.show()
        utils.log("Now Starting test!", self.sys_conf.model_name, True)
        if self.is_pkl:
            net = torch.load(self.test_path, map_location=self.sys_conf.device_in_prog)
        else:
            net = self.__get_model()
            para = torch.load(self.test_path, map_location=self.sys_conf.device_in_prog)
            net.load_state_dict(para)
        net.eval()
        net.to(self.sys_conf.device_in_prog)
        test_data = utils.Data(self.sys_conf, self.data_config, False)
        test_data.update_scale(self.curr_scale)
        self.is_normal = test_data.normal
        psnr_list = []
        ssim_list = []
        dataset_list = []
        for dataset in self.sys_conf.test_dataset:
            self.curr_dataset = dataset
            dataset_list.append(dataset)
            utils.log("Now testing dataset is " + self.curr_dataset + " at scale " + str(self.curr_scale), self.sys_conf.model_name, True)
            test_data.update_dataset(self.curr_dataset)
            if self.sys_conf.model_mode == "post":
                if self.sys_conf.scale_pos == "init":
                    psnr, ssim = self.__test_pre_or_init(net, test_data.get_loader(), self.is_normal)
                elif self.sys_conf.scale_pos == "forward":
                    psnr, ssim = self.__test_scale_pos_is_forward(net, test_data.get_loader(), self.is_normal)
                else:
                    raise NameError("WRONG MODEL SCALE POSITION!")
            elif self.sys_conf.model_mode == "pre":
                psnr, ssim = self.__test_pre_or_init(net, test_data.get_loader(), test_data.normal)
            else:
                raise NameError("WRONG MODEL MODE!")
            psnr_list.append(psnr)
            ssim_list.append(ssim)
            utils.log("PSNR: " + str(round(psnr, 2)) + "db", self.sys_conf.model_name, True)
            utils.log("SSIM: " + str(round(ssim, 4)), self.sys_conf.model_name, True)
        utils.make_csv_file_at_test_once(psnr_list, ssim_list, dataset_list, self.sys_conf.test_file, self.save_path, self.sys_conf.test_color_channel)
    def __test_all(self):
        self.show()
        for dataset in self.sys_conf.test_dataset:
            self.curr_dataset = dataset
            utils.log("Now testing dataset " + self.curr_dataset, self.sys_conf.model_name, True)
            for scale in self.sys_conf.scale_factor:
                self.curr_scale = scale
                utils.log("Now testing scale " + str(self.curr_scale), self.sys_conf.model_name, True)
                if self.sys_conf.shave_is_scale:
                    self.sys_conf.shave = self.curr_scale
                test_data = utils.Data(self.sys_conf, self.data_config, False)
                test_data.update_dataset(self.curr_dataset)
                test_data.update_scale(self.curr_scale)
                self.is_normal = test_data.normal
                total_model_num = self.sys_conf.Epoch // self.sys_conf.save_step
                net = self.__get_model()
                net.eval()
                psnr_list = numpy.zeros(total_model_num + 1, dtype=float)
                ssim_list = numpy.zeros(total_model_num + 1, dtype=float)
                for i in range(1, total_model_num + 1):
                    test_file = "net_x" + str(self.curr_scale) + "_" + str(i * self.sys_conf.save_step) + ".pth"
                    utils.log("Now testing model " + str(test_file), self.sys_conf.model_name, True)
                    test_path = self.test_path + test_file
                    para = torch.load(test_path, map_location=self.sys_conf.device_in_prog)
                    net.load_state_dict(para)
                    net.to(self.sys_conf.device_in_prog)
                    if self.sys_conf.model_mode == "post":
                        if self.sys_conf.scale_pos == "init":
                            psnr, ssim = self.__test_pre_or_init(net, test_data.get_loader(), self.is_normal)
                        elif self.sys_conf.scale_pos == "forward":
                            psnr, ssim = self.__test_scale_pos_is_forward(net, test_data.get_loader(), self.is_normal)
                        else:
                            raise NameError("WRONG MODEL SCALE POSITION!")
                    elif self.sys_conf.model_mode == "pre":
                        psnr, ssim = self.__test_pre_or_init(net, test_data.get_loader(), self.is_normal)
                    else:
                        raise NameError("WRONG MODEL MODE!")
                    psnr_list[i] = psnr
                    ssim_list[i] = ssim
                if self.sys_conf.drew:
                    utils.drew_pic(psnr_list, ssim_list, self.curr_dataset, self.curr_scale, self.sys_conf.save_step, self.save_path, self.sys_conf.test_color_channel)
                utils.make_csv_file(psnr_list, ssim_list, self.curr_dataset, self.curr_scale, self.sys_conf.save_step, self.save_path, self.sys_conf.test_color_channel)
    def show(self):
        utils.log("", self.sys_conf.model_name)
        utils.log("--------This is model test config-------", self.sys_conf.model_name)
        utils.log("Test dataset: " + str(self.sys_conf.test_dataset), self.sys_conf.model_name)
        utils.log("Test scale: " + str(self.sys_conf.scale_factor), self.sys_conf.model_name) 
        utils.log("Test color channel: " + self.sys_conf.test_color_channel, self.sys_conf.model_name)
        utils.log("Shave: " + ("the same as scale" if self.sys_conf.shave_is_scale else str(self.sys_conf.shave)), self.sys_conf.model_name)
        utils.log("Test mode: " + ("All" if self.sys_conf.test_all else "Once"), self.sys_conf.model_name)
        if not self.sys_conf.test_all:
            utils.log("Test file: " + self.sys_conf.test_file, self.sys_conf.model_name)
    def __test_scale_pos_is_forward(self, net, loader, is_normal):
        mean_psnr = 0.0
        mean_ssim = 0.0
        test_len = 0
        for _, data in enumerate(loader):
            test_len += 1
            lr, hr = data
            sr = net(lr.to(self.sys_conf.device_in_prog), self.curr_scale)
            sr = sr.permute(0, 2, 3, 1).squeeze(0).cpu()
            hr = hr.permute(0, 2, 3, 1).squeeze(0).cpu()
            if is_normal:
                sr = sr * 255.
                hr = hr * 255.
            psnr, ssim = utils.prepare_and_compute(sr, hr, self.sys_conf.shave, self.sys_conf.test_color_channel)
            mean_psnr += psnr
            mean_ssim += ssim
        mean_psnr = mean_psnr / test_len
        mean_ssim = mean_ssim / test_len
        return mean_psnr, mean_ssim
    def __test_pre_or_init(self, net, loader, is_normal):
        mean_psnr = 0.0
        mean_ssim = 0.0
        test_len = 0
        for _, data in enumerate(loader):
            test_len += 1
            lr, hr = data
            sr = net(lr.to(self.sys_conf.device_in_prog))
            sr = sr.permute(0, 2, 3, 1).squeeze(0).cpu()
            hr = hr.permute(0, 2, 3, 1).squeeze(0).cpu()
            if is_normal:
                sr = sr * 255.
                hr = hr * 255.
            psnr, ssim = utils.prepare_and_compute(sr, hr, self.sys_conf.shave, self.sys_conf.test_color_channel)
            mean_psnr += psnr
            mean_ssim += ssim
        mean_psnr = mean_psnr / test_len
        mean_ssim = mean_ssim / test_len
        return mean_psnr, mean_ssim
    def test(self):
        if self.sys_conf.test_all:
            self.__test_all()
        else:
            self.__test_once()