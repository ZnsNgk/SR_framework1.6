import numpy
import cv2
import math
import pandas
import matplotlib.pyplot as plt

def rgb2ycbcr(rgb): #opencv读取的通道顺序实际上是BGR
    m = numpy.array([[24.966, 128.553, 65.481],
                  [112, -74.203, -37.797],
                  [-18.214, -93.786, 112]])
    shape = rgb.shape
    if len(shape) == 3:
        rgb = rgb.reshape((shape[0] * shape[1], 3))
    ycbcr = numpy.dot(rgb, m.transpose() / 255.)
    ycbcr[:,0] += 16.
    ycbcr[:,1:] += 128.
    return ycbcr.reshape(shape)

def compute_psnr(img1, img2):
    diff = (img1/1.0 - img2/1.0) ** 2
    mse = float(numpy.mean(diff))
    if mse < 1.0e-10:
       return 100
    return 10. * math.log10(255.0**2/mse)

def compute_ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(numpy.float64)
    img2 = img2.astype(numpy.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = numpy.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

def calculate_ssim(img1, img2):
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return compute_ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(compute_ssim(img1, img2))
            return numpy.array(ssims).mean()
        elif img1.shape[2] == 1:
            return compute_ssim(numpy.squeeze(img1), numpy.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')

def shave(hr, sr, shave_pix):
    shave_pix = math.ceil(shave_pix)
    if len(sr.shape) == 2:
        sr_new = sr[shave_pix:-shave_pix, shave_pix:-shave_pix]
        hr_new = hr[shave_pix:-shave_pix, shave_pix:-shave_pix]
    elif len(sr.shape) == 3:
        sr_new = sr[shave_pix:-shave_pix, shave_pix:-shave_pix, :]
        hr_new = hr[shave_pix:-shave_pix, shave_pix:-shave_pix, :]
    else:
        raise ValueError("Wrong Dimension!")
    return hr_new, sr_new

def prepare_and_compute(sr, hr, shave_pix, is_Y):
    sr = numpy.array(sr)
    hr = numpy.array(hr)
    _, _, c = sr.shape
    if is_Y == "Y":
        if c == 3:
            sr = rgb2ycbcr(sr)
            sr = sr[:,:,0]
            hr = rgb2ycbcr(hr)
            hr = hr[:,:,0]
        elif c == 1:
            sr = sr[:,:,0]
            hr = hr[:,:,0]
        h_h,w_h = hr.shape
        h_s,w_s = sr.shape
        if h_h < h_s:
            sr = sr[0:h_h,:]
        else:
            hr = hr[0:h_s,:]
        if w_h < w_s:
            sr = sr[:,0:w_h]
        else:
            hr = hr[:,0:w_s]
    elif is_Y == "RGB":
        if c == 3:
            h_h,w_h, _ = hr.shape
            h_s,w_s, _ = sr.shape
            if h_h < h_s:
                sr = sr[0:h_h,:,:]
            else:
                hr = hr[0:h_s,:,:]
            if w_h < w_s:
                sr = sr[:,0:w_h,:]
            else:
                hr = hr[:,0:w_s,:]
        elif c == 1:
            raise NameError("WRONG TEST COLOR CHANNEL!")
    if shave_pix > 0:
        hr, sr = shave(hr, sr, shave_pix)
    psnr = compute_psnr(sr, hr)
    ssim = calculate_ssim(sr, hr)
    return psnr, ssim

def drew_pic(psnr_list, ssim_list, dataset_name, scale, step, result_folder, color_channel):
    length_psnr = len(psnr_list)
    length_ssim = len(ssim_list)
    if length_ssim != length_psnr:
        raise ValueError("The length of PSNR_list array and SSIM_list array should be equal")
    length = length_psnr
    x = numpy.arange(0, length*step, step)
    plt.figure(figsize = (14,7))
    plt.suptitle('Scale='+str(scale)+', Dataset is '+ dataset_name + ", Color is " + color_channel)
    plt.subplot(121)
    psnr_argmax = numpy.argmax(psnr_list)
    psnr_list[0] = psnr_list[psnr_argmax]
    plt.xlim(xmax = length*step, xmin = step)
    plt.plot(x, psnr_list, label='PSNR', color='b')
    plt.title('PSNR Max is ' + str(round(psnr_list[psnr_argmax],2)) + ' db' + ', position is:' + str(psnr_argmax*step))
    plt.xlabel('Epoch')
    plt.ylabel('PSNR')
    plt.legend()
    plt.grid(color="k", linestyle=":")
    plt.subplot(122)
    ssim_argmax = numpy.argmax(ssim_list)
    ssim_list[0] = ssim_list[ssim_argmax]
    plt.xlim(xmax = length*step, xmin = step)
    plt.plot(x, ssim_list, label='SSIM', color='b')
    plt.title('SSIM Max is ' + str(round(ssim_list[ssim_argmax],4)) + ', position is:' + str(ssim_argmax*step))
    plt.xlabel('Epoch')
    plt.ylabel('SSIM')
    plt.legend()
    plt.grid(color="k", linestyle=":")
    plt.savefig(result_folder + dataset_name + '_' + str(scale) + "_" + color_channel + '.png', dpi = 600)
    plt.close()
    
def make_csv_file(psnr_list, ssim_list, dataset_name, scale, step, result_folder, color_channel):
    length_psnr = len(psnr_list)
    length_ssim = len(ssim_list)
    if length_ssim != length_psnr:
        raise ValueError("The length of PSNR_list array and SSIM_list array should be equal")
    length = length_psnr
    psnr_list = psnr_list[1:]
    ssim_list = ssim_list[1:]
    step_list = numpy.arange(step, step * length, step)
    result_list = numpy.zeros([length-1, 3], dtype='float32')
    result_list[:, 0] = step_list[:]
    result_list[:, 1] = psnr_list[:]
    result_list[:, 2] = ssim_list[:]
    csv_data = pandas.DataFrame(result_list, columns=['Epoch', 'PSNR', 'SSIM'])
    csv_data.to_csv(result_folder + dataset_name + '_' + str(scale) + "_" + color_channel + '.csv', encoding='utf-8', index=False)

def make_csv_file_at_test_once(psnr_list, ssim_list, dataset_list, test_file, result_folder, color_channel):
    length_psnr = len(psnr_list)
    length_ssim = len(ssim_list)
    length_dataset = len(dataset_list)
    if length_ssim != length_psnr and length_dataset != length_psnr:
        raise ValueError("The length of PSNR_list array, SSIM_list array and Dataset_list array should be equal")
    str_psnr_list = []
    str_ssim_list = []
    for psnr in psnr_list:
        str_psnr_list.append(str(round(psnr, 2)))
    for ssim in ssim_list:
        str_ssim_list.append(str(round(ssim, 4)))
    result_list = [[] for i in range(3)]
    for dataset in dataset_list:
        result_list[0].append(dataset)
    for psnr in str_psnr_list:
        result_list[1].append(psnr)
    for ssim in str_ssim_list:
        result_list[2].append(ssim)
    result_list = numpy.array(result_list, dtype=str)
    result_list = result_list.transpose(1, 0)
    csv_data = pandas.DataFrame(result_list, columns=['Dataset', 'PSNR', 'SSIM'])
    csv_data.to_csv(result_folder + test_file.replace('.pth', '').replace('.pkl', '') + "_" + color_channel + '.csv', encoding='utf-8', index=False)