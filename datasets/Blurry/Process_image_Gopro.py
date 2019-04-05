from pdb import set_trace as st
import os
import numpy as np
import cv2
import argparse

parser = argparse.ArgumentParser('create image pairs')
parser.add_argument('--fold_A', dest='fold_A', help='input directory for image A', type=str, default='blur')
parser.add_argument('--fold_B', dest='fold_B', help='input directory for image B', type=str, default='sharp')
parser.add_argument('--num_imgs', dest='num_imgs', help='number of images',type=int, default=1000000)
parser.add_argument('--use_AB', dest='use_AB', help='if true: (0001_A, 0001_B) to (0001_AB)',action='store_true')
parser.add_argument('--phase', dest='phase', help='/test or /train', type=str, default='./test')
parser.add_argument('--fine_size', dest='fine_size', type=int, default=720, help='then crop to this size')
parser.add_argument('--img_width', dest='img_width', type=int, default=1280, help='image width')
args = parser.parse_args()

# set output directory
img_fold_AB = args.phase + '_01'
if not os.path.isdir(img_fold_AB):
    os.makedirs(img_fold_AB)

# folders in /test or /train
splits = os.listdir(args.phase)

# fine_size and image width
fine_size = args.fine_size
img_width = args.img_width

counter = 0

for folder in splits:
    if not folder.startswith('.'):
        img_fold_A = os.path.join(args.phase, folder, args.fold_A)
        img_fold_B = os.path.join(args.phase, folder, args.fold_B)

        img_list = os.listdir(img_fold_A)
        num_imgs = min(args.num_imgs, len(img_list))

        counter += 1
        for n in range(num_imgs):
            name_A = img_list[n]
            name_B = name_A

            prefix = '%02d' % counter
            name_AB = prefix + name_A

            path_A = os.path.join(img_fold_A, name_A)
            path_B = os.path.join(img_fold_B, name_B)
            path_AB = os.path.join(img_fold_AB, name_AB)

            im_A = cv2.imread(path_A, cv2.IMREAD_COLOR)
            im_B = cv2.imread(path_B, cv2.IMREAD_COLOR)

            #w1 = int(np.ceil(np.random.uniform(1e-2, img_width - fine_size)))
            #im_A = im_A[0:0 + fine_size, w1:w1 + fine_size]
            #im_B = im_B[0:0 + fine_size, w1:w1 + fine_size]

            im_A = cv2.resize(im_A, (fine_size,fine_size))
            im_B = cv2.resize(im_B, (fine_size, fine_size))

            im_AB = np.concatenate([im_A, im_B], 1)
            cv2.imwrite(path_AB, im_AB)


img_list_AB = os.listdir(img_fold_AB)
print('The total image number is %d.' % (len(img_list_AB)))




