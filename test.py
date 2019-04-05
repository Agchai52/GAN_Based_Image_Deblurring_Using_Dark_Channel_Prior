from __future__ import division
import os
import time
from glob import glob
import tensorflow as tf
import numpy as np
from six.moves import xrange

from ops import *
from utils import *


def test(self, args):
    """Test DeblurGAn"""
    init_op = tf.global_variables_initializer()
    self.sess.run(init_op)

    if self.load_for_test(self.checkpoint_dir):
        print(" [*] Load SUCCESS")
    else:
        print(" [!] Load failed...")

    sample_files = glob('./datasets/test/*.png')

    # sort testing input
    n = [int(i) for i in map(lambda x: x.split('/')[-1].split('.png')[0], sample_files)]
    sample_files = [x for (y, x) in sorted(zip(n, sample_files))]

    # load testing input
    print("Loading testing images ...")

    sample = [load_data(sample_file, is_test=True, fine_size=self.image_size) for sample_file in sample_files]

    if (self.is_grayscale):
        sample_images = np.array(sample).astype(np.float32)[:, :, :, None]
    else:
        sample_images = np.array(sample).astype(np.float32)

    sample_images = [sample_images[i:i + self.batch_size]
                     for i in xrange(0, len(sample_images), self.batch_size)]
    sample_images = np.array(sample_images)
    print(sample_images.shape)

    for i, sample_image in enumerate(sample_images):
        idx = i + 1
        print("sampling image ", idx)
        samples = self.sess.run(
            self.fake_B_sample,
            feed_dict={self.real_data: sample_image}
        )
        save_images(samples, [self.batch_size, 1],
                    './{}/test_{:04d}.png'.format(args.test_dir, idx))
