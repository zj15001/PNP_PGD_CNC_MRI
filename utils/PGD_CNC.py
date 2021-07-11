"""
The definition of PGD-CNC algorithm
Authors:JinCheng Li  (201971138@yangtzeu.edu.cn)
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import os.path
import cv2
import logging
import argparse
import numpy as np
from collections import OrderedDict
import scipy.io as sio
import torch
from utils import utils_logger
from utils import utils_image as util
from utils.utils import Df


def soft(x, c):
    return np.fmax(np.fabs(x) - c, 0) * np.sign(x)

def analyze_parse_PGD_CNC(default_alpha, default_iter_num,default_lambda1,default_b):
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, default=default_alpha, help="Step size in Plug-and Play")
    parser.add_argument("--iter_num", type=int, default=default_iter_num, help="Number of iterations")
    parser.add_argument("--lambda1", type=float, default=default_lambda1, help="regularization parameter")
    parser.add_argument("--b", type=float, default=default_b, help="convex parameter") # Here b is b^2 in the original article
    PGD_CNC_opt = parser.parse_args()
    return PGD_CNC_opt

def PGD_CNC(mask, noises, **PGD_CNC_opts):

    # ----------------------------------------
    # Preparation
    # ----------------------------------------

    iter_num = PGD_CNC_opts.get('iter_num', 50)
    alpha = PGD_CNC_opts.get('alpha', 0.4)
    lambda1 = PGD_CNC_opts.get('lambda1', 0.05)
    b = PGD_CNC_opts.get('b', 36)     # Here b is b^2 in the original article
    task_current = 'dn'  # 'dn' for denoising
    testset_name = 'Set1'  # test set,set1
    n_channels = 1
    sf = 1  # unused for denoising
    show_img = False  # default: False
    save_E = True  # save estimated image
    save_LEH = False  # save zoomed LR, E and H images
    border = 0
    A = np.zeros((256, 256), dtype='uint8')
    out = [A, A, A, A, A, A, A, A, A, A, A, A, A, A, A]
    psnr1 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    n = 0
    use_clip = True
    testsets = 'testsets'  # fixed
    results = 'results'  # fixed
    result_name = testset_name + '_' + task_current + 'PGD CNC'
    torch.cuda.empty_cache()

    # ----------------------------------------
    # L_path, E_path, H_path
    # ----------------------------------------
    L_path = os.path.join(testsets, testset_name)  # L_path, for Low-quality images
    E_path = os.path.join(results, result_name)  # E_path, for Estimated images
    util.mkdir(E_path)

    logger_name = result_name
    utils_logger.logger_info(logger_name, log_path=os.path.join(E_path, logger_name + '.log'))
    logger = logging.getLogger(logger_name)

    test_results = OrderedDict()
    test_results['psnr'] = []
    test_results['ssim'] = []
    test_results['re'] = []

    test_results_ave = OrderedDict()
    test_results_ave['psnr'] = []
    test_results_ave['ssim'] = []
    test_results_ave['re'] = []

    logger.info(L_path)
    L_paths = util.get_image_paths(L_path)

    for idx, img in enumerate(L_paths):

        # --------------------------------
        # (1) get img_L
        # --------------------------------

        img_name, ext = os.path.splitext(os.path.basename(img))
        img_H = util.imread_uint(img, n_channels=n_channels)
        img_H = util.modcrop(img_H, 8)
        img_L = util.uint2single(img_H)

        if use_clip:
            img_L = util.uint2single(util.single2uint(img_L))
        util.imshow(img_L) if show_img else None

        # --------------------------------
        # (2) initialize x
        # --------------------------------

        img_L = img_L.squeeze()
        y = np.fft.fft2(img_L) * mask + noises  # observed value
        img_L_init = np.fft.ifft2(y)  # zero fill
        print("(zero-filling) psnr = %.4f" % util.psnr(img_L_init*255, img_L*255))
        x = np.copy(img_L_init)

        # --------------------------------
        # (3) main iterations
        # --------------------------------

        for i in range(iter_num):

            x = np.real(x)
            z = x - alpha * Df(x, mask, y)
            z = np.absolute(z)

            c1 = 1 / b
            p = soft(x, c1)
            s = x - p

            c2 = alpha * lambda1
            t = z + alpha * lambda1 * b * s
            x = soft(t, c2)

        # --------------------------------
        # (4) img_E
        # --------------------------------
        out[n] = x
        img_E = x * 255

        if n_channels == 1:
            img_H = img_H.squeeze()
        if save_E:
            util.imsave(img_E, os.path.join(E_path, img_name + '_PDG CNC.png'))

        # --------------------------------
        # (5) img_LEH
        # --------------------------------
        if save_LEH:
            img_L = util.single2uint(img_L)
            k_v = k / np.max(k) * 1.0
            k_v = util.single2uint(np.tile(k_v[..., np.newaxis], [1, 1, 3]))
            k_v = cv2.resize(k_v, (3 * k_v.shape[1], 3 * k_v.shape[0]), interpolation=cv2.INTER_NEAREST)
            img_I = cv2.resize(img_L, (sf * img_L.shape[1], sf * img_L.shape[0]), interpolation=cv2.INTER_NEAREST)
            img_I[:k_v.shape[0], -k_v.shape[1]:, :] = k_v
            img_I[:img_L.shape[0], :img_L.shape[1], :] = img_L
            util.imshow(np.concatenate([img_I, img_E, img_H], axis=1),
                        title='LR / Recovered / Ground-truth') if show_img else None
            util.imsave(np.concatenate([img_I, img_E, img_H], axis=1),
                        os.path.join(E_path, img_name + '_k' + '_LEH.png'))

        psnr = util.calculate_psnr(img_E, img_H, border=border)
        ssim = util.calculate_ssim(img_E, img_H, border=border)
        re = util.calculate_re(img_E, img_H, border=border)
        test_results['psnr'].append(psnr)
        test_results['ssim'].append(ssim)
        test_results['re'].append(re)
        logger.info('{:s} - PSNR: {:.4f} dB; SSIM: {:.4f} ; RE: {:.4f}.'.format(img_name + ext, psnr, ssim, re))
        util.imshow(np.concatenate([img_E, img_H], axis=1),
                    title='Recovered / Ground-truth') if show_img else None
        psnr1[n] = psnr
        n += 1

    # --------------------------------
    # Average PSNR
    # --------------------------------

    ave_psnr = sum(test_results['psnr']) / len(test_results['psnr'])
    ave_ssim = sum(test_results['ssim']) / len(test_results['ssim'])
    ave_re = sum(test_results['re']) / len(test_results['re'])
    logger.info('------> testset_name: ({}), Average PSNR:({:.3f})dB, Average ssim : ({:.3f}), Average re : ({:.3f}) )'.format(
            testset_name, ave_psnr, ave_ssim, ave_re))
    test_results_ave['psnr'].append(ave_psnr)
    test_results_ave['ssim'].append(ave_ssim)
    test_results_ave['re'].append(ave_re)

    return out, psnr1
