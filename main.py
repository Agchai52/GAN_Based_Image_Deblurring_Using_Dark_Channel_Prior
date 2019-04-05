import argparse
import os
import scipy.misc
import numpy as np
# from model import pix2pix
from networks import model_pix2pix
import tensorflow as tf
import train
import test
import test_psnr

parser = argparse.ArgumentParser(description='')
parser.add_argument('--dataset_name', dest='dataset_name', default='Blurry', help='name of the dataset, facades for pix2pix, Blurry for DeblurGAN')
parser.add_argument('--epoch', dest='epoch', type=int, default=15, help='# of epoch')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=1, help='# images in batch')
parser.add_argument('--train_size', dest='train_size', type=int, default=1e8, help='# images used to train')
parser.add_argument('--load_scale', dest='load_scale', type=float, default=1.0, help='scale images to this size')
parser.add_argument('--fine_sizeX', dest='fine_size', type=int, default=256, help='then crop to this size')
parser.add_argument('--ngf', dest='ngf', type=int, default=64, help='# of gen filters in first conv layer')
parser.add_argument('--ndf', dest='ndf', type=int, default=64, help='# of discri filters in first conv layer')
parser.add_argument('--input_nc', dest='input_nc', type=int, default=3, help='# of input image channels')
parser.add_argument('--output_nc', dest='output_nc', type=int, default=3, help='# of output image channels')
parser.add_argument('--niter', dest='niter', type=int, default=200, help='# of iter at starting learning rate')
parser.add_argument('--lr', dest='lr', type=float, default=0.0002, help='initial learning rate for adam')
parser.add_argument('--beta1', dest='beta1', type=float, default=0.5, help='momentum term of adam')
parser.add_argument('--flip', dest='flip', type=bool, default=True, help='if flip the images for data argumentation')
parser.add_argument('--add_noise', dest='add_noise', type=bool, default=True, help='if add noise on the images for data argumentation')
parser.add_argument('--which_direction', dest='which_direction', default='AtoB', help='AtoB or BtoA')
parser.add_argument('--phase', dest='phase', default='train', help='train, test')
parser.add_argument('--save_epoch_freq', dest='save_epoch_freq', type=int, default=4, help='save a model every save_epoch_freq epochs (does not overwrite previously saved models)')
parser.add_argument('--save_latest_freq', dest='save_latest_freq', type=int, default=5000, help='save the latest model every latest_freq sgd iterations (overwrites the previous latest model)')
parser.add_argument('--print_freq', dest='print_freq', type=int, default=50, help='print the debug information every print_freq iterations')
parser.add_argument('--continue_train', dest='continue_train', type=bool, default=False, help='if continue training, load the latest model: 1: true, 0: false')
parser.add_argument('--serial_batches', dest='serial_batches', type=bool, default=False, help='f 1, takes images in order to make batches, otherwise takes them randomly')
parser.add_argument('--serial_batch_iter', dest='serial_batch_iter', type=bool, default=True, help='iter into serial image list')
parser.add_argument('--checkpoint_dir', dest='checkpoint_dir', default='./checkpoint', help='models are saved here')
parser.add_argument('--sample_dir', dest='sample_dir', default='./sample', help='sample are saved here')
parser.add_argument('--test_dir', dest='test_dir', default='./test', help='test sample are saved here')
parser.add_argument('--test_psnr', dest='test_psnr', type=bool, default=False, help='skip saving test images and only compute psnr')
parser.add_argument('--L1_lambda', dest='L1_lambda', type=float, default=100.0, help='weight on L1 term in objective')
parser.add_argument('--dark_channel_lambda', dest='dark_channel_lambda', type=float, default=250.0, help='weight on Dark Channel loss in objective')
parser.add_argument('--gpu', '-g', default=-1, type=int, help='GPU ID (negative value indicates CPU)')

args = parser.parse_args()

def main(_):
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
    if not os.path.exists(args.sample_dir):
        os.makedirs(args.sample_dir)
    if not os.path.exists(args.test_dir):
        os.makedirs(args.test_dir)

    if args.gpu > -1:
        device = '/gpu:{}'.format(args.gpu)
    else:
        device = '/cpu:0'
    config = tf.ConfigProto(allow_soft_placement=True)
    with tf.device(device):
        with tf.Session(config=config) as sess:
            model = model_pix2pix(sess, image_size=args.fine_size, batch_size=args.batch_size,
                                  output_size=args.fine_size,
                                  dark_channel_lambda=args.dark_channel_lambda, dataset_name=args.dataset_name,
                                  checkpoint_dir=args.checkpoint_dir, sample_dir=args.sample_dir, phase=args.phase)

            if args.phase == 'train':
                train.train(model, args)
            else:
                if (args.test_psnr==False):
                    test.test(model, args)
                else:
                    test_psnr.test(model, args)


if __name__ == '__main__':
    tf.app.run()
