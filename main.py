"""
Plug-and-Play Algorithm for Convex-Nonconvex Sparse Regularization with Application to MRI Reconstruction
Authors:JinCheng Li  (201971138@yangtzeu.edu.cn)
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch
import cv2
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from utils.PGD_L1 import PGD_L1
from utils.PNP_PGD_L1_BM3D import PNP_PGD_L1_BM3D
from utils.PNP_PGD_L1_D import PNP_PGD_L1_D
from utils.PGD_CNC import PGD_CNC
from utils.PNP_PGD_CNC_BM3D import PNP_PGD_CNC_BM3D
from utils.PNP_PGD_CNC_D import PNP_PGD_CNC_DnCNN
from utils.PNP_PGD_CNC_D import PNP_PGD_CNC_D

from utils.PGD_L1 import analyze_parse_PGD_L1
from utils.PNP_PGD_L1_BM3D import analyze_parse_PNP_PGD_L1_BM3D
from utils.PNP_PGD_L1_D import analyze_parse_PNP_PGD_L1_D
from utils.PGD_CNC import analyze_parse_PGD_CNC
from utils.PNP_PGD_CNC_BM3D import analyze_parse_PNP_PGD_CNC_BM3D
from utils.PNP_PGD_CNC_D import analyze_parse_PNP_PGD_CNC_DnCNN
from utils.PNP_PGD_CNC_D import analyze_parse_PNP_PGD_CNC_D
from utils.utils  import enlargement

def sub_plot_org(n, img, M, N):
    plt.subplot(M, N, n)
    plt.imshow(img, cmap = plt.cm.gray)
    plt.axis('off')

def sub_plot_res(num, a, psnr, M, N):
    plt.subplot(M, N, num)
    xx= cv2.putText(a, '%.2f dB' % psnr, (160, 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (1, 1, 1), 2, cv2.LINE_AA)
    plt.imshow(xx, cmap = plt.cm.gray)
    plt.axis('off')

def sub_plot_small(n, x_noises,MM, NN):
    plt.subplot(MM, NN, n)
    plt.imshow(x_noises, cmap=plt.cm.gray)
    plt.axis('off')

def sub_plot_error_img(num, X, min, j, M ,N):
    """
    This function is used to plot error images of three denoising methods.
    [min, max] is the relative error range of pixels.
    """
    s = [0.2, 0.2, 0.2]
    max = s[j]
    plt.subplot(M ,N, num)
    plt.imshow(X, vmin = min, vmax = max, cmap = plt.cm.gray)
    plt.axis('off')

def sub_plot_error_img_small(num, X, min, j, M, N):
    """
    This function is used to plot error images of three denoising methods.
    [min, max] is the relative error range of pixels.
    """
    s = [0.2, 0.2, 0.2]
    max = s[j]
    plt.subplot(M, N, num)
    plt.imshow(X, vmin = min, vmax = max, cmap = plt.cm.gray)
    plt.axis('off')


# ---- input arguments ----
" PGD-L1 "
PGD_L1_opt = analyze_parse_PGD_L1(0.4, 50, 0.05)
#the arguments are default_alpha, default_max iteration，default_lambda1

" PNP-PGD-L1 "        " 1  BM3D "
PNP_PGD_L1_BM3D_opt = analyze_parse_PNP_PGD_L1_BM3D(0.4, 50)
# the arguments are default sigma, default alpha and default max iteration.

" PNP-PGD-L1 "      " 2  neural network "
PNP_PGD_L1_D_opt1 = analyze_parse_PNP_PGD_L1_D(0.4, 50)   #dncnn, fdncnny, ircnn, ffdnet, drunet
# # the arguments are default sigma, default alpha and default max iteration.
PNP_PGD_L1_D_opt2 = analyze_parse_PNP_PGD_L1_D(0.25, 50)   #IRCNN
# the arguments are default sigma, default alpha and default max iteration.

" PGD-CNC "
PGD_CNC_opt = analyze_parse_PGD_CNC(0.4, 50, 0.05, 36)
# the arguments are default_alpha, default_max iteration, default_lambda1, default_b

" PNP-PGD-CNC "     " 1  BM3D "
PNP_PGD_CNC_BM3D_opt = analyze_parse_PNP_PGD_CNC_BM3D(0.4, 50, 1.7, 1)
# the arguments are default_alpha, default_max iteration, default_lambda1, default_b

" PNP-PGD-CNC "     " 2  neural network "
PNP_PGD_CNC_D_opt1 = analyze_parse_PNP_PGD_CNC_D(0.4, 50, 1, 1)   #FDnCNN
# the arguments are default_alpha, default_max iteration, deufault_lambda1, default_b
PNP_PGD_CNC_DnCNN_opt = analyze_parse_PNP_PGD_CNC_DnCNN(0.4, 50, 2, 1)   #DnCNN
# the arguments are default_alpha, default_max iteration, default_lambda1, default_b
PNP_PGD_CNC_D_opt3 = analyze_parse_PNP_PGD_CNC_D(0.4, 50, 1.45, 1)  #FFDNet
# # the arguments are default_alpha, default_max iteration, default_lambda1, default_b
PNP_PGD_CNC_D_opt4 = analyze_parse_PNP_PGD_CNC_D(0.25, 50, 2.5, 1)   #IRCNN
# the arguments are default_alpha, default_max iteration, default_lambda1, default_b
PNP_PGD_CNC_D_opt5 = analyze_parse_PNP_PGD_CNC_D(0.4, 50, 2.25, 1)  #DRUNet
# # the arguments are default_alpha, default_max iteration, default_lambda1, default_b

j, k = 0, 0  # j = im_orig number, k = mask number

with torch.no_grad():

    # ---- load mask matrix ----
    mat = np.array([sio.loadmat('CS_MRI/Q_Random30.mat'),
                    sio.loadmat('CS_MRI/Q_Radial30.mat'),
                    sio.loadmat('CS_MRI/Q_Cartesian30.mat')])
    mask = np.array([mat[0].get('Q1').astype(np.float64),
                     mat[1].get('Q1').astype(np.float64),
                     mat[2].get('Q1').astype(np.float64)])
    mask1 = np.fft.fftshift(mask)

    # ---- load noises -----
    noises = sio.loadmat('CS_MRI/noises.mat')
    noises = noises.get('noises').astype(np.complex128) * 3.0

    # ---- set options -----
    PGD_L1_opts = dict(alpha=PGD_L1_opt.alpha, iter_num=PGD_L1_opt.iter_num, lambda1=PGD_L1_opt.lambda1)
    PNP_PGD_L1_BM3D_opts = dict(alpha=PNP_PGD_L1_BM3D_opt.alpha, iter_num=PNP_PGD_L1_BM3D_opt.iter_num)
    PNP_PGD_L1_D_opts1 = dict(alpha=PNP_PGD_L1_D_opt1.alpha, iter_num=PNP_PGD_L1_D_opt1.iter_num)
    PNP_PGD_L1_D_opts2 = dict(alpha=PNP_PGD_L1_D_opt2.alpha, iter_num=PNP_PGD_L1_D_opt2.iter_num)
    PGD_CNC_opts = dict(alpha=PGD_CNC_opt.alpha, iter_num=PGD_CNC_opt.iter_num,
                             lambda1=PGD_CNC_opt.lambda1, b=PGD_CNC_opt.b)
    PNP_PGD_CNC_BM3D_opts = dict(alpha=PNP_PGD_CNC_BM3D_opt.alpha, iter_num=PNP_PGD_CNC_BM3D_opt.iter_num,
                             lambda1=PNP_PGD_CNC_BM3D_opt.lambda1, b=PNP_PGD_CNC_BM3D_opt.b)
    PNP_PGD_CNC_D_opts1 = dict(alpha=PNP_PGD_CNC_D_opt1.alpha, iter_num=PNP_PGD_CNC_D_opt1.iter_num,
                             lambda1=PNP_PGD_CNC_D_opt1.lambda1, b=PNP_PGD_CNC_D_opt1.b)
    PNP_PGD_CNC_DnCNN_opts = dict(alpha=PNP_PGD_CNC_DnCNN_opt.alpha, iter_num=PNP_PGD_CNC_DnCNN_opt.iter_num,
                             lambda1=PNP_PGD_CNC_DnCNN_opt.lambda1, b=PNP_PGD_CNC_DnCNN_opt.b)
    PNP_PGD_CNC_D_opts3 = dict(alpha=PNP_PGD_CNC_D_opt3.alpha, iter_num=PNP_PGD_CNC_D_opt3.iter_num,
                             lambda1=PNP_PGD_CNC_D_opt3.lambda1, b=PNP_PGD_CNC_D_opt3.b)
    PNP_PGD_CNC_D_opts4 = dict(alpha=PNP_PGD_CNC_D_opt4.alpha, iter_num=PNP_PGD_CNC_D_opt4.iter_num,
                             lambda1=PNP_PGD_CNC_D_opt4.lambda1, b=PNP_PGD_CNC_D_opt4.b)
    PNP_PGD_CNC_D_opts5 = dict(alpha=PNP_PGD_CNC_D_opt5.alpha, iter_num=PNP_PGD_CNC_D_opt5.iter_num,
                             lambda1=PNP_PGD_CNC_D_opt5.lambda1, b=PNP_PGD_CNC_D_opt5.b)

    #  load demo synthetic block image and demo noisy image
    im_orig = np.array([cv2.imread('testsets/set/11.png', 0) / 255.0])

    # ---- denoising work -----
    name = ['fdncnn_gray', 'dncnn_15', 'ffdnet_gray', 'ircnn_gray', 'drunet_gray', 'dncnn_25', 'dncnn_50']

    " PGD-L1 "
    out1, psnr1 = PGD_L1(mask[k], noises, **PGD_L1_opts)
    out1 = out1[j]
    psnr1 = psnr1[j]

    " PNP-PGD-L1 "        " 1  BM3D "
    out2, psnr2 = PNP_PGD_L1_BM3D(mask[k], **PNP_PGD_L1_BM3D_opts)
    out2 = out2[j]
    psnr2 = psnr2[j]

    " PNP-PGD-L1 "      " 2  neural network "
    out3, psnr3 = PNP_PGD_L1_D(name[0], mask[k], noises, **PNP_PGD_L1_D_opts1)
    out3 = out3[j]
    psnr3 = psnr3[j]

    out4, psnr4 = PNP_PGD_L1_D(name[1], mask[k], noises, **PNP_PGD_L1_D_opts1)
    out4 = out4[j]
    psnr4 = psnr4[j]

    out5, psnr5 = PNP_PGD_L1_D(name[2], mask[k], noises, **PNP_PGD_L1_D_opts1)
    out5 = out5[j]
    psnr5 = psnr5[j]

    out6, psnr6 = PNP_PGD_L1_D(name[3], mask[k], noises, **PNP_PGD_L1_D_opts2)
    out6 = out6[j]
    psnr6 = psnr6[j]

    out7, psnr7 = PNP_PGD_L1_D(name[4], mask[k], noises, **PNP_PGD_L1_D_opts1)
    out7 = out7[j]
    psnr7 = psnr7[j]

    " PGD-CNC "
    out8, psnr8 = PGD_CNC(mask[k], noises, **PGD_CNC_opts)
    out8 = out8[j]
    psnr8 = psnr8[j]

    " PNP-PGD-CNC "     " 1  BM3D "
    out9, psnr9 = PNP_PGD_CNC_BM3D(mask[k], **PNP_PGD_CNC_BM3D_opts)
    out9 = out9[j]
    psnr9 = psnr9[j]

    " PNP-PGD-CNC "     " 2  neural network "
    out10, psnr10 = PNP_PGD_CNC_D(name[0], mask[k], noises, **PNP_PGD_CNC_D_opts1)
    out10 = out10[j]
    psnr10 = psnr10[j]

    out11, psnr11 = PNP_PGD_CNC_DnCNN(name[5], name[1],  mask[k], noises, **PNP_PGD_CNC_DnCNN_opts)
    out11 = out11[j]
    psnr11 = psnr11[j]

    out12, psnr12 = PNP_PGD_CNC_D(name[2], mask[k], noises, **PNP_PGD_CNC_D_opts3)
    out12 = out12[j]
    psnr12 = psnr12[j]

    out13, psnr13 = PNP_PGD_CNC_D(name[3], mask[k], noises, **PNP_PGD_CNC_D_opts4)
    out13 = out13[j]
    psnr13 = psnr13[j]

    out14, psnr14 = PNP_PGD_CNC_D(name[4], mask[k], noises, **PNP_PGD_CNC_D_opts5)
    out14 = out14[j]
    psnr14 = psnr14[j]

    ''' plot demo result figure '''
    region = [[40, 80],
              [90, 80],
              [40, 130],
              [90, 130]]

    region1 = [[150, 175],
              [200, 175],
              [150, 225],
              [200, 225]]

    plt.figure(1)
    plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95,
                        wspace=0.05, hspace=0.05)
    x1 = enlargement(j, out1, region)
    x2 = enlargement(j, out2, region)
    x3 = enlargement(j, out3, region)
    x4 = enlargement(j, out4, region)
    x5 = enlargement(j, out5, region)
    x6 = enlargement(j, out6, region)
    x7 = enlargement(j, out7, region)
    x1a = enlargement(j, x1[0], region1)
    x2a = enlargement(j, x2[0], region1)
    x3a = enlargement(j, x3[0], region1)
    x4a = enlargement(j, x4[0], region1)
    x5a = enlargement(j, x5[0], region1)
    x6a = enlargement(j, x6[0], region1)
    x7a = enlargement(j, x7[0], region1)
    sub_plot_res(1, x1a[0], psnr1, 2, 7)
    sub_plot_res(2, x2a[0], psnr2, 2, 7)
    sub_plot_res(3, x3a[0], psnr3, 2, 7)
    sub_plot_res(4, x4a[0], psnr4, 2, 7)
    sub_plot_res(5, x5a[0], psnr5, 2, 7)
    sub_plot_res(6, x6a[0], psnr6, 2, 7)
    sub_plot_res(7, x7a[0], psnr7, 2, 7)
    sub_plot_small(29, x1[1], 4, 14)
    sub_plot_small(30, x1a[1], 4, 14)
    sub_plot_small(31, x2[1], 4, 14)
    sub_plot_small(32, x2a[1], 4, 14)
    sub_plot_small(33, x3[1], 4, 14)
    sub_plot_small(34, x3a[1], 4, 14)
    sub_plot_small(35, x4[1], 4, 14)
    sub_plot_small(36, x4a[1], 4, 14)
    sub_plot_small(37, x5[1], 4, 14)
    sub_plot_small(38, x5a[1], 4, 14)
    sub_plot_small(39, x6[1], 4, 14)
    sub_plot_small(40, x6a[1], 4, 14)
    sub_plot_small(41, x7[1], 4, 14)
    sub_plot_small(42, x7a[1], 4, 14)

    plt.figure(2)
    plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95,
                        wspace=0.05, hspace=0.05)
    out1A = np.fabs(out1 - im_orig[j])
    out2A = np.fabs(out2 - im_orig[j])
    out3A = np.fabs(out3 - im_orig[j])
    out4A = np.fabs(out4 - im_orig[j])
    out5A = np.fabs(out5 - im_orig[j])
    out6A = np.fabs(out6 - im_orig[j])
    out7A = np.fabs(out7 - im_orig[j])
    out1A = enlargement(j, out1A, region)
    out2A = enlargement(j, out2A, region)
    out3A = enlargement(j, out3A, region)
    out4A = enlargement(j, out4A, region)
    out5A = enlargement(j, out5A, region)
    out6A = enlargement(j, out6A, region)
    out7A = enlargement(j, out7A, region)
    out1AA = enlargement(j, out1A[0], region1)
    out2AA = enlargement(j, out2A[0], region1)
    out3AA = enlargement(j, out3A[0], region1)
    out4AA = enlargement(j, out4A[0], region1)
    out5AA = enlargement(j, out5A[0], region1)
    out6AA = enlargement(j, out6A[0], region1)
    out7AA = enlargement(j, out7A[0], region1)
    sub_plot_error_img(1, out1AA[0], 0, j,  2, 7)
    sub_plot_error_img(2, out2AA[0], 0, j,  2, 7)
    sub_plot_error_img(3, out3AA[0], 0, j,  2, 7)
    sub_plot_error_img(4, out4AA[0], 0, j,  2, 7)
    sub_plot_error_img(5, out5AA[0], 0, j,  2, 7)
    sub_plot_error_img(6, out6AA[0], 0, j,  2, 7)
    sub_plot_error_img(7, out7AA[0], 0, j,  2, 7)
    sub_plot_error_img_small(29, out1A[1], 0, j, 4, 14)
    sub_plot_error_img_small(30, out1AA[1], 0, j, 4, 14)
    sub_plot_error_img_small(31, out2A[1], 0, j, 4, 14)
    sub_plot_error_img_small(32, out2AA[1], 0, j, 4, 14)
    sub_plot_error_img_small(33, out3A[1], 0, j, 4, 14)
    sub_plot_error_img_small(34, out3AA[1], 0, j, 4, 14)
    sub_plot_error_img_small(35, out4A[1], 0, j, 4, 14)
    sub_plot_error_img_small(36, out4AA[1], 0, j, 4, 14)
    sub_plot_error_img_small(37, out5A[1], 0, j, 4, 14)
    sub_plot_error_img_small(38, out5AA[1], 0, j, 4, 14)
    sub_plot_error_img_small(39, out6A[1], 0, j, 4, 14)
    sub_plot_error_img_small(40, out6AA[1], 0, j, 4, 14)
    sub_plot_error_img_small(41, out7A[1], 0, j, 4, 14)
    sub_plot_error_img_small(42, out7AA[1], 0, j, 4, 14)

    plt.figure(3)
    plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95,
                        wspace=0.05, hspace=0.05)
    x8 = enlargement(j, out8, region)
    x9 = enlargement(j, out9, region)
    x10 = enlargement(j, out10, region)
    x11 = enlargement(j, out11, region)
    x12 = enlargement(j, out12, region)
    x13 = enlargement(j, out13, region)
    x14 = enlargement(j, out14, region)
    x8a = enlargement(j, x8[0], region1)
    x9a = enlargement(j, x9[0], region1)
    x10a = enlargement(j, x10[0], region1)
    x11a = enlargement(j, x11[0], region1)
    x12a = enlargement(j, x12[0], region1)
    x13a = enlargement(j, x13[0], region1)
    x14a = enlargement(j, x14[0], region1)
    sub_plot_res(1, x8a[0], psnr8, 2, 7)
    sub_plot_res(2, x9a[0], psnr9, 2, 7)
    sub_plot_res(3, x10a[0], psnr10, 2, 7)
    sub_plot_res(4, x11a[0], psnr11, 2, 7)
    sub_plot_res(5, x12a[0], psnr12, 2, 7)
    sub_plot_res(6, x13a[0], psnr13, 2, 7)
    sub_plot_res(7, x14a[0], psnr14, 2, 7)
    sub_plot_small(29, x8[1], 4, 14)
    sub_plot_small(30, x8a[1], 4, 14)
    sub_plot_small(31, x9[1], 4, 14)
    sub_plot_small(32, x9a[1], 4, 14)
    sub_plot_small(33, x10[1], 4, 14)
    sub_plot_small(34, x10a[1], 4, 14)
    sub_plot_small(35, x11[1], 4, 14)
    sub_plot_small(36, x11a[1], 4, 14)
    sub_plot_small(37, x12[1], 4, 14)
    sub_plot_small(38, x12a[1], 4, 14)
    sub_plot_small(39, x13[1], 4, 14)
    sub_plot_small(40, x13a[1], 4, 14)
    sub_plot_small(41, x14[1], 4, 14)
    sub_plot_small(42, x14a[1], 4, 14)


    plt.figure(4)
    plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95,
                        wspace=0.05, hspace=0.05)
    out8A = np.fabs(out8 - im_orig[j])
    out9A = np.fabs(out9 - im_orig[j])
    out10A = np.fabs(out10 - im_orig[j])
    out11A = np.fabs(out11 - im_orig[j])
    out12A = np.fabs(out12 - im_orig[j])
    out13A = np.fabs(out13 - im_orig[j])
    out14A = np.fabs(out14 - im_orig[j])
    out8A = enlargement(j, out8A, region)
    out9A = enlargement(j, out9A, region)
    out10A = enlargement(j, out10A, region)
    out11A = enlargement(j, out11A, region)
    out12A = enlargement(j, out12A, region)
    out13A = enlargement(j, out13A, region)
    out14A = enlargement(j, out14A, region)
    out8AA = enlargement(j, out8A[0], region1)
    out9AA = enlargement(j, out9A[0], region1)
    out10AA = enlargement(j, out10A[0], region1)
    out11AA = enlargement(j, out11A[0], region1)
    out12AA = enlargement(j, out12A[0], region1)
    out13AA = enlargement(j, out13A[0], region1)
    out14AA = enlargement(j, out14A[0], region1)
    sub_plot_error_img(1, out8AA[0], 0, j,  2, 7)
    sub_plot_error_img(2, out9AA[0], 0, j,  2, 7)
    sub_plot_error_img(3, out10AA[0], 0, j,  2, 7)
    sub_plot_error_img(4, out11AA[0], 0, j,  2, 7)
    sub_plot_error_img(5, out12AA[0], 0, j,  2, 7)
    sub_plot_error_img(6, out13AA[0], 0, j,  2, 7)
    sub_plot_error_img(7, out14AA[0], 0, j,  2, 7)
    sub_plot_error_img_small(29, out8A[1], 0, j, 4, 14)
    sub_plot_error_img_small(30, out8AA[1], 0, j, 4, 14)
    sub_plot_error_img_small(31, out9A[1], 0, j, 4, 14)
    sub_plot_error_img_small(32, out9AA[1], 0, j, 4, 14)
    sub_plot_error_img_small(33, out10A[1], 0, j, 4, 14)
    sub_plot_error_img_small(34, out10AA[1], 0, j, 4, 14)
    sub_plot_error_img_small(35, out11A[1], 0, j, 4, 14)
    sub_plot_error_img_small(36, out11AA[1], 0, j, 4, 14)
    sub_plot_error_img_small(37, out12A[1], 0, j, 4, 14)
    sub_plot_error_img_small(38, out12AA[1], 0, j, 4, 14)
    sub_plot_error_img_small(39, out13A[1], 0, j, 4, 14)
    sub_plot_error_img_small(40, out13AA[1], 0, j, 4, 14)
    sub_plot_error_img_small(41, out14A[1], 0, j, 4, 14)
    sub_plot_error_img_small(42, out14AA[1], 0, j, 4, 14)

    plt.rcParams['savefig.dpi'] = 600  # image pixel
    plt.rcParams['figure.dpi'] = 600  # resolution ratio
    plt.show()