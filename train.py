from __future__ import division
import os
import time
from glob import glob
import tensorflow as tf
import numpy as np
from six.moves import xrange
import cv2

from ops import *
from utils import *

def train(self, args):
    """Train pix2pix"""
    d_optim = tf.train.AdamOptimizer(learning_rate=args.lr, beta1=args.beta1) \
        .minimize(self.d_loss, var_list=self.d_vars)
    g_optim = tf.train.AdamOptimizer(learning_rate=args.lr, beta1=args.beta1) \
        .minimize(self.g_loss, var_list=self.g_vars)

    init_op = tf.global_variables_initializer()
    self.sess.run(init_op)

    self.g_sum = tf.summary.merge([self.d__sum,
                                   self.fake_B_sum, self.d_loss_fake_sum, self.g_loss_sum])
    self.d_sum = tf.summary.merge([self.d_sum, self.d_loss_real_sum, self.d_loss_sum])
    self.writer = tf.summary.FileWriter("./logs", self.sess.graph)

    loss_record = "loss_record.txt"

    counter = 1
    start_time = time.time()

    if self.load(self.checkpoint_dir):
        print(" [*] Load SUCCESS")
    else:
        print(" [!] Load failed...")

    data = glob('./datasets/train/*.png'.format(self.dataset_name))
    print(len(data))

    for epoch in xrange(args.epoch):
        data = glob('./datasets/train/*.png'.format(self.dataset_name))
        # np.random.shuffle(data)
        batch_idxs = min(len(data), args.train_size) // self.batch_size

        for idx in xrange(0, batch_idxs):
            batch_files = data[idx * self.batch_size:(idx + 1) * self.batch_size]
            batch = [load_data(batch_file, fine_size=self.image_size) for batch_file in batch_files]
            if (self.is_grayscale):
                batch_images = np.array(batch).astype(np.float32)[:, :, :, None]
            else:
                batch_images = np.array(batch).astype(np.float32)

            # Update D network
            _, summary_str = self.sess.run([d_optim, self.d_sum],
                                           feed_dict={self.real_data: batch_images})
            self.writer.add_summary(summary_str, counter)

            # Update G network
            _, summary_str = self.sess.run([g_optim, self.g_sum],
                                           feed_dict={self.real_data: batch_images})
            self.writer.add_summary(summary_str, counter)

            # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
            _, summary_str = self.sess.run([g_optim, self.g_sum],
                                           feed_dict={self.real_data: batch_images})
            self.writer.add_summary(summary_str, counter)

            errD_fake = self.d_loss_fake.eval({self.real_data: batch_images})
            errD_real = self.d_loss_real.eval({self.real_data: batch_images})
            errG = self.g_loss.eval({self.real_data: batch_images})
            errDC = self.dark_channel_loss.eval({self.real_data: batch_images})
            real_A = self.real_A.eval({self.real_data: batch_images})
            counter += 1
            print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f, dark_channel_loss: %.8f" \
                  % (epoch, idx, batch_idxs,
                     time.time() - start_time, errD_fake + errD_real, errG, errDC))

            # img_dark_channel = self.dark_channel(self.real_B).eval({self.real_data: batch_images})
            #show_image(real_A, [self.batch_size, 1], "Dark Channel")
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()

            # To record losses in a .txt file
            losses_dg = [errD_fake + errD_real, errG, errDC]
            losses_dg_str = " ".join(str(v) for v in losses_dg)

            with open(loss_record, 'a+') as file:
                file.writelines(losses_dg_str + "\n")

            if np.mod(counter, args.save_latest_freq) == 50:
                self.save(args.checkpoint_dir, counter)

        if np.mod(epoch, args.save_epoch_freq) == 3:
            self.save(args.checkpoint_dir, counter)

    #loss_record_wobj.close()