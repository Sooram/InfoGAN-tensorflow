# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 10:01:30 2019

@author: Sooram Kang

Reference: https://github.com/AndyHsiao26/InfoGAN-Tensorflow

"""

import argparse
import tensorflow as tf
from infogan import InfoGAN, train, inference


parser = argparse.ArgumentParser(description='InfoGAN')
parser.add_argument('--model_dir', type=str, 
                      default='../exp',
                      help='Directory in which the model is stored')
#parser.add_argument('--data_dir', type=str,
#                      default='C:\\Users\\CHANG\\PycharmProjects\\InfoGAN\\data',
#                      help='Directory in which the data is stored')
parser.add_argument('--is_training', type=bool, default=False, help='whether it is training or inferecing')
parser.add_argument('--fix_var', type=bool, default=False, help='whether to approximate variance')
parser.add_argument('--num_category', type=int, default=10, help='category dim of latent variable')
parser.add_argument('--num_cont', type=int, default=3, help='continuous dim of latent variable')
parser.add_argument('--num_rand', type=int, default=62, help='random noise dim of latent variable')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--epoch', type=int, default=5000, help='epochs')
parser.add_argument('--d_lr', type=float, default=2e-4, help='learning rate for discriminator')
parser.add_argument('--g_lr', type=float, default=1e-3, help='learning rate for generator')


        
#%%
def main(args):
    # build model 
    model = InfoGAN(args)
    
    # open session 
    c = tf.ConfigProto()
    c.gpu_options.visible_device_list = "0"
    
    sess = tf.Session(config=c)
    sess.run(tf.global_variables_initializer())

    train(args, model, sess) if args.is_training else inference(args, model, sess)


    
if __name__ == '__main__':
    config, unparsed = parser.parse_known_args()
    main(config)







