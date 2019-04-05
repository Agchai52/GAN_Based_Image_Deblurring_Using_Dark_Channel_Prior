from pdb import set_trace as st
import os
import numpy as np
import cv2
import argparse

parser = argparse.ArgumentParser('create image pairs')
parser.add_argument('--fold_A', dest='fold_A', help='input directory for image A', type=str, default='./trainA')
parser.add_argument('--fold_B', dest='fold_B', help='input directory for image B', type=str, default='./trainB')
parser.add_argument('--fold_AB', dest='fold_AB', help='output directory', type=str, default='./train_AB')
parser.add_argument('--num_imgs', dest='num_imgs', help='number of images',type=int, default=1000000)
parser.add_argument('--use_AB', dest='use_AB', help='if true: (0001_A, 0001_B) to (0001_AB)',action='store_false')
args = parser.parse_args()

for arg in vars(args):
    print('[%s] = ' % arg,  getattr(args, arg))

sp = '1'


img_fold_A = args.fold_A
img_fold_B = args.fold_B
img_list = os.listdir(img_fold_A)
print('Image number in trainA is %d' % (len(img_list)))
print('Image number in trainB is %d' % (len(os.listdir(img_fold_B))))

num_imgs = min(int(args.num_imgs), len(img_list))
print('split = %s, use %d/%d images' % (sp, num_imgs, len(img_list)))
img_fold_AB = args.fold_AB
if not os.path.isdir(img_fold_AB):
    os.makedirs(img_fold_AB)
print('split = %s, number of images = %d' % (sp, num_imgs))
for n in range(num_imgs):
    name_A = img_list[n]
    path_A = os.path.join(img_fold_A, name_A)
    if args.use_AB:
        name_B = name_A.replace('_A.', '_B.')
    else:
        name_B = name_A
    path_B = os.path.join(img_fold_B, name_B)
    if os.path.isfile(path_A) and os.path.isfile(path_B):
        name_AB = name_A
        if args.use_AB:
            name_AB = name_AB.replace('_A.', '.') # remove _A
        path_AB = os.path.join(img_fold_AB, name_AB)
        im_A = cv2.imread(path_A, cv2.IMREAD_COLOR)
        im_B = cv2.imread(path_B, cv2.IMREAD_COLOR)
        try:
            im_AB = np.concatenate([im_A, im_B], 1)
        except ValueError:
            if im_A is None:
                print("Oops! The shape of imagA is NoneType. Name of A is %s" % (path_A))
            elif im_B is None:
                print("Oops! The shape of imagB is NoneType. Name of B is %s" % (path_B))
            continue
        cv2.imwrite(path_AB, im_AB)
img_list_AB = os.listdir(img_fold_AB)
print('Image number in trainAB is %d' % (len(img_list_AB)))
