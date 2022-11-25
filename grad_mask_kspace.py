# -*- coding: utf-8 -*-
"""
Graduate k-space masking for MRI image denoising and blurring
(for Agilent FID data)

Created on Sun Nov 20 2022
Last modified on Fri Nov 25 2022

@author: Beata Wereszczyńska
"""
import nmrglue as ng
import numpy as np
import matplotlib.pyplot as plt
import cv2

def grad_mask_kspace(path, number_of_slices, picked_slice, r):
    """
    Graduate k-space masking for MRI image denoising and blurring
    (for Agilent FID data).
    Input:
        .fid folder location: path [str],
        total number of slices in the MRI experiment: number_of_slices [int],
        selected slice number: picked_slice [int],
        radius for k-space masking in pixels: r [int].
    """

    # import k-space data
    echoes = ng.agilent.read(dir=path)[1]
    kspace = echoes[picked_slice - 1 : echoes.shape[0] : number_of_slices, :]  # downsampling to one slice
    del path, echoes, number_of_slices, picked_slice
    
    # k-space masking
    mask_denoise = np.zeros(shape=kspace.shape)
    mask = np.zeros(shape=kspace.shape)
    for value in range(r, r+15, 2):
        cv2.circle(img=mask, center=(int(kspace.shape[0]/2),int(kspace.shape[1]/2)), 
                    radius = value, color =(1,0,0), thickness=-1)
        mask_denoise = mask_denoise + mask
    masked_k = np.multiply(kspace, mask_denoise)
    del mask, value, mask_denoise
    
    # normalization
    masked_k = masked_k / (np.max(abs(masked_k)) / np.max(abs(kspace)))
    
    # reconstruct the original image
    ft1 = np.fft.fft2(kspace)                 # 2D FFT
    ft1 = np.fft.fftshift(ft1)                # fixing problem with corner being center of the image
    ft1 = np.transpose(np.flip(ft1, (1,0)))   # matching geometry with VnmrJ-calculated image (still a bit shifted)
    
    # reconstruct denoised image
    ft2 = np.fft.fft2(masked_k)               # 2D FFT
    ft2 = np.fft.fftshift(ft2)                # fixing problem with corner being center of the image
    ft2 = np.transpose(np.flip(ft2, (1,0)))   # matching geometry with VnmrJ-calculated image (still a bit shifted)
    
    # visualization
    plt.rcParams['figure.dpi'] = 600
    plt.subplot(141)
    plt.title('Original k-space', fontdict = {'fontsize' : 7}), plt.axis('off')
    plt.imshow(abs(kspace), cmap=plt.get_cmap('gray'), vmax=int(np.mean(abs(kspace))*7))
    plt.subplot(142)
    plt.title(f'Masked k-space (r = {r})', fontdict = {'fontsize' : 7}), plt.axis('off')
    plt.imshow(abs(masked_k), cmap=plt.get_cmap('gray'), vmax=int(np.mean(abs(kspace))*7))
    plt.subplot(143)
    plt.title('Original image', fontdict = {'fontsize' : 7}), plt.axis('off')
    plt.imshow(abs(ft1), cmap=plt.get_cmap('gray'))
    plt.subplot(144)
    plt.title('New image', fontdict = {'fontsize' : 7}), plt.axis('off')
    plt.imshow(abs(ft2), cmap=plt.get_cmap('gray'))
    plt.tight_layout(pad=0, w_pad=0.2, h_pad=0)
    plt.show()
    del r
    
    # return data
    return kspace, masked_k, ft1, ft2


def main():
    path = 'mems_20190406_02.fid'     # .fid folder location [str]
    number_of_slices = 384            # total number of slices in the imaging experiment [int]
    picked_slice = 116                # selected slice number [int]
    r = 56                            # radius for k-space masking in pixels [int]


    # running calculations and retrieving the results
    k, km, ft1, ft2 = grad_mask_kspace(path, number_of_slices, picked_slice, r)
    
    # creating global variables to be available after the run completion
    global MRI_k
    MRI_k = k
    global MRI_km
    MRI_km = km
    global MRI_ft1
    MRI_ft1 = ft1
    global MRI_ft2
    MRI_ft2 = ft2


if __name__ == "__main__":
    main()    
