from pdb import set_trace as st
import os
import numpy as np
import cv2
import argparse
from glob import glob
from skimage.util import random_noise as imnoise

parser = argparse.ArgumentParser('create image pairs')
parser.add_argument('--output_fold', dest='output_fold', help='output directory for aligned image left blur right sharp', type=str, default='aligned')
parser.add_argument('--fold_B', dest='fold_B', help='ouput directory for image B', type=str, default='sharp')
parser.add_argument('--num_imgs', dest='num_imgs', help='number of images',type=int, default=110000000)
parser.add_argument('--phase', dest='phase', help='test or train', type=str, default='train')
parser.add_argument('--fine_size', dest='fine_size', type=int, default=720, help='then crop to this size')
args = parser.parse_args()

# set frames in each blurry image
N = 11

# set input directory
img_input_fold = './' + args.phase
# set output directory
img_output_fold = img_input_fold + '_' + args.output_fold
if not os.path.isdir(img_output_fold):
    os.makedirs(img_output_fold)

# folders in /test or /train
splits = os.listdir(img_input_fold)

# fine_size and image width
fine_size = args.fine_size

counter = 0

for folder in splits:
    if not folder.startswith('.'):
        img_subset = os.path.join(img_input_fold, folder)

        img_list = os.listdir(img_subset)
        img_list = sorted(img_list, key=str.lower)
        num_imgs = min(args.num_imgs, len(img_list))

        out_num, extra_num = divmod(num_imgs, N)

        counter += 1
        prefix = '%02d' % counter

        for n in range(out_num):
            img_in = []
            for i in range(N):
                name_single = img_list[n*N + i]

                path_in = os.path.join(img_subset, name_single)

                im_single = cv2.imread(path_in, cv2.IMREAD_COLOR).astype(np.float)

                im_noise = imnoise(im_single/255, mode='gaussian', clip=True, var=0.001)

                im_noise = cv2.resize(im_noise, (fine_size, fine_size))

                img_in.append(im_noise)

            img_in = np.array(img_in)

            img_blur = np.mean(img_in, axis=0)
            img_sharp = img_in[5]

            img_AB = np.concatenate([img_blur, img_sharp], 1)

            name_align = prefix + '%04d' % (n+1)
            path_out = os.path.join(img_output_fold, name_align)
            path_out = path_out + '.png'
            print(name_align)
            cv2.imwrite(path_out, img_AB*255)






