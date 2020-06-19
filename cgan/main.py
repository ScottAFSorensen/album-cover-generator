import os
from CGAN import CGAN

from utils import show_all_variables
from utils import check_folder

import tensorflow as tf
import argparse

"""parsing and configuration"""
def parse_args():
    desc = "Tensorflow implementation of GAN collections"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--gan_type', type=str, default='CGAN',
                        choices=['CGAN', 'GAN', 'infoGAN', 'ACGAN', 'EBGAN', 'BEGAN', 'WGAN', 'WGAN_GP', 'DRAGAN', 'LSGAN', 'VAE', 'CVAE'],
                        help='The type of GAN', required=True)
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'fashion-mnist', 'celebA'],
                        help='The name of dataset')
    parser.add_argument('--epoch', type=int, default=20, help='The number of epochs to run')
    parser.add_argument('--batch_size', type=int, default=32, help='The size of batch')
    parser.add_argument('--z_dim', type=int, default=64, help='Dimension of noise vector')
    parser.add_argument('--checkpoint_dir', type=str, default='cpoint',
                        help='Directory name to save the checkpoints')
    parser.add_argument('--result_dir', type=str, default='result',
                        help='Directory name to save the generated images')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Directory name to save training logs')

    return check_args(parser.parse_args())

"""checking arguments"""
def check_args(args):
    # --checkpoint_dir
    check_folder(args.checkpoint_dir)

    # --result_dir
    check_folder(args.result_dir)

    # --result_dir
    check_folder(args.log_dir)

    # --epoch
    assert args.epoch >= 1, 'number of epochs must be larger than or equal to one'

    # --batch_size
    assert args.batch_size >= 1, 'batch size must be larger than or equal to one'

    # --z_dim
    assert args.z_dim >= 1, 'dimension of noise vector must be larger than or equal to one'

    return args

"""main"""
def main():
    

    path = './results'
    
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        

        gan = None

        gan = CGAN(sess, epoch=100000, batch_size=32, z_dim=64, dataset_name='mnist', checkpoint_dir=str(path+'/cpoint'), result_dir=str(path+'/result'), log_dir=str(path+'/training'))
        if gan is None:
            raise Exception("[!] There is no option for " + args.gan_type)

        # build graph
        gan.build_model()

        # show network architecture
        show_all_variables()

        # launch the graph in a session
        gan.train()
        print(" [*] Training finished!")

        # visualize learned generator
        gan.visualize_results(100000-1)

        print(" [*] Testing finished!")

if __name__ == '__main__':
    main()
