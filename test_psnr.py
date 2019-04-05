from __future__ import division
import os
import time
from glob import glob
import tensorflow as tf
import numpy as np
from six.moves import xrange
import math
from skimage.measure import compare_ssim as ssim

from ops import *
from utils import *

def psnr(img1, img2):
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def test(self, args):
    """Test DeblurGAn"""
    init_op = tf.global_variables_initializer()
    self.sess.run(init_op)

    if self.load_for_test(self.checkpoint_dir):
        print(" [*] Load SUCCESS")
    else:
        print(" [!] Load failed...")

    test_data = glob('./datasets/test/*.png')

    # sort testing input
    n = [int(i) for i in map(lambda x: x.split('/')[-1].split('.png')[0], test_data)]
    test_data = [x for (y, x) in sorted(zip(n, test_data))]

    print(len(test_data))

    psnr_n = 0.0
    ssim_n = 0.0
    counter = 0

    for idx in xrange(0, len(test_data)):
        # load testing input
        sample_files = test_data[idx * self.batch_size:(idx + 1) * self.batch_size]
        sample = [load_data(sample_file, is_test=True, fine_size=self.image_size) for sample_file in sample_files]

        sample_images = np.array(sample).astype(np.float32)

        print("sampling image ", idx+1)
        samples = self.sess.run(
            self.fake_B_sample,
            feed_dict={self.real_data: sample_images})

        # == This part is for computing PSNR and SSIM
        img_sharp = sample_images[0][:, :, 3:6]
        img_deblu = samples[0]

        img_sharp = inverse_transform(img_sharp) * 255
        img_deblu = inverse_transform(img_deblu) * 255

        psnr_n += psnr(img_deblu, img_sharp)
        ssim_n += ssim(img_deblu / 255, img_sharp / 255, gaussian_weights=True, multichannel=True,use_sample_covariance=False)
        counter += 1

    print(psnr_n / counter)
    print(ssim_n / counter)
    # == This part is for computing PSNR and SSIM

