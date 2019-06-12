# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 09:34:55 2019

@author: Sooram Kang

Reference: https://github.com/AndyHsiao26/InfoGAN-Tensorflow

"""
import os
import cv2
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.layers as layers
from tensorflow.examples.tutorials.mnist import input_data
from general_model import Model

img = {
       'w': 28,
       'h': 28,
       'c': 1
       }
       

class InfoGAN(Model):
    def __init__(self, args):

        #########################
        #                       #
        #    General Setting    #
        #                       #
        #########################

        self.args = args

        self.model_dir = args.model_dir

        if not self.model_dir:
            raise ValueError('Need to provide model directory')

        self.log_dir = os.path.join(self.model_dir, 'log')
        self.test_dir = os.path.join(self.model_dir, 'test')

        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        if not os.path.exists(self.test_dir):
            os.makedirs(self.test_dir)

        self.global_step = tf.train.get_or_create_global_step()
        #########################
        #                       #
        #     Model Building    #
        #                       #
        #########################
                
        net_G = Generator()
        net_D = Discriminator()

        # 1. Build Generator

        # Create latent variable
        with tf.name_scope('noise_sample'):
            self.z_cat = tf.placeholder(tf.int32, [None])
            self.z_cont = tf.placeholder(tf.float32, [None, args.num_cont])
            self.z_rand = tf.placeholder(tf.float32, [None, args.num_rand])

            z = tf.concat([tf.one_hot(self.z_cat, args.num_category), self.z_cont, self.z_rand], axis=1)

        self.g = net_G(z, args)
        
        # 2. Build Discriminator

        # Real Data
        with tf.name_scope('data_and_target'):
            self.x = tf.placeholder(tf.float32, [None, img['w'], img['h'], img['c']])

            y_real = tf.ones([tf.shape(self.x)[0]])
            y_fake = tf.zeros([tf.shape(self.x)[0]])

        d_real, _, _, _ = net_D(self.x, args)
        d_fake, r_cat, r_cont_mu, r_cont_var = net_D(self.g, args)

        # 3. Calculate loss

        # -log(D(G(x))) trick
        with tf.name_scope('loss'):
            self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fake, labels=y_real))

            self.d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fake, labels=y_fake))
            self.d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_real, labels=y_real))

            self.d_loss = (self.d_loss_fake + self.d_loss_real)
            
            # discrete logQ(c|x)
            self.cat_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=r_cat,
                                                                                          labels=self.z_cat))

            eplison = (r_cont_mu-self.z_cont) / r_cont_var

            # variance = 1
            # log gaussian distribution (continuous logQ(c|x))
            self.cont_loss = -tf.reduce_mean(tf.reduce_sum(-0.5*tf.log(2*np.pi*r_cont_var+1e-8)-0.5*tf.square(eplison), axis=1))

            self.train_g_loss = self.g_loss + self.cat_loss + self.cont_loss*0.1
            self.train_d_loss = self.d_loss + self.cat_loss + self.cont_loss*0.1

        # 4. Update weights
        g_param = tf.trainable_variables(scope='generator')
        d_param = tf.trainable_variables(scope='discriminator')

        with tf.name_scope('optimizer'):
            g_optim = tf.train.AdamOptimizer(learning_rate=args.g_lr, beta1=0.5, beta2=0.99)
            self.g_train_op = g_optim.minimize(self.train_g_loss, var_list=g_param, global_step=self.global_step)
            d_optim = tf.train.AdamOptimizer(learning_rate=args.d_lr, beta1=0.5, beta2=0.99)
            self.d_train_op = d_optim.minimize(self.train_d_loss, var_list=d_param)

            
        # 5. Visualize
        tf.summary.image('Real', self.x)
        tf.summary.image('Fake', self.g)

        with tf.name_scope('Generator'):
            tf.summary.scalar('g_total_loss', self.train_g_loss)
        with tf.name_scope('Discriminator'):
            tf.summary.scalar('d_total_loss', self.train_d_loss)
        with tf.name_scope('All_Loss'):
            tf.summary.scalar('g_loss', self.g_loss)
            tf.summary.scalar('d_loss', self.d_loss)
            tf.summary.scalar('cat_loss', self.cat_loss)
            tf.summary.scalar('cont_loss', self.cont_loss)

        self.summary_op = tf.summary.merge_all()
        
        super(InfoGAN, self).__init__(5)


class Generator(object):
    def __call__(self, inputs, params):
        with slim.arg_scope([layers.fully_connected, layers.conv2d_transpose],
                            activation_fn=tf.nn.relu, normalizer_fn=layers.batch_norm,
                            normalizer_params={'updates_collections': None, 'is_training': params.is_training, 'decay': 0.9}):
            with tf.variable_scope('generator', reuse=tf.AUTO_REUSE):
                net = layers.fully_connected(inputs, 1024)
                net = layers.fully_connected(net, 7*7*128)
                net = tf.reshape(net, [-1, 7, 7, 128])
                net = layers.conv2d_transpose(net, 64, [4, 4], stride=2)
                net = layers.conv2d_transpose(net, 1, [4, 4], stride=2, normalizer_fn=None, activation_fn=tf.nn.sigmoid)
        
        return net
                
class Discriminator(object):
    def __call__(self, inputs, params):
        with slim.arg_scope([layers.fully_connected, layers.conv2d],
                            activation_fn=self.leaky_relu, normalizer_fn=None,
                            normalizer_params={'updates_collections': None, 'is_training': params.is_training, 'decay': 0.9}):
            with tf.variable_scope('discriminator', reuse=tf.AUTO_REUSE):
                with tf.variable_scope('shared'):
                    d1 = layers.conv2d(inputs, 64, [4, 4], stride=2)
                    d2 = layers.conv2d(d1, 128, [4, 4], stride=2, normalizer_fn=layers.batch_norm)
    
                    d2_flatten = layers.flatten(d2)
    
                    d3 = layers.fully_connected(d2_flatten, 1024, normalizer_fn=layers.batch_norm)
    
                with tf.variable_scope('d'):
                    d_out = layers.fully_connected(d3, 1, activation_fn=None)
                    d_out = tf.squeeze(d_out, axis=-1)
    
                with tf.variable_scope('q'):
                    r1 = layers.fully_connected(d3, 128, normalizer_fn=layers.batch_norm)
                    r_cat = layers.fully_connected(r1, params.num_category, activation_fn=None)
                    r_cont_mu = layers.fully_connected(r1, params.num_cont, activation_fn=None)
                    if params.fix_var:
                        r_cont_var = 1
                    else:
                        r_cont_logvar = layers.fully_connected(r1, params.num_cont, activation_fn=None)
                        r_cont_var = tf.exp(r_cont_logvar)
                         
                return d_out, r_cat, r_cont_mu, r_cont_var

    def leaky_relu(self, x):
        return tf.where(tf.greater(x, 0), x, 0.1 * x)
        
#%%
def train(args, model, sess):
    summary_writer = tf.summary.FileWriter(model.log_dir, sess.graph)
    
    # load data
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    
    # load previous model
    model.load(sess, args.model_dir)  

    steps_per_epoch = mnist.train.labels.shape[0] // args.batch_size

    for epoch in range(args.epoch):
        for step in range(steps_per_epoch):
            x_batch, _ = mnist.train.next_batch(args.batch_size)
            x_batch = np.reshape(x_batch, [-1, img['w'], img['h'], img['c']])
            z_cont = np.random.uniform(-1, 1, size=[args.batch_size, args.num_cont])
            z_rand = np.random.uniform(-1, 1, size=[args.batch_size, args.num_rand])
            z_cat = np.random.randint(args.num_category, size=[args.batch_size])

            d_loss, _ = sess.run([model.train_d_loss, model.d_train_op], 
                                 feed_dict={model.x: x_batch,
                                           model.z_cat: z_cat,
                                           model.z_cont: z_cont,
                                           model.z_rand: z_rand})

            g_loss, _ = sess.run([model.train_g_loss, model.g_train_op],
                                 feed_dict={model.x: x_batch,
                                            model.z_cat: z_cat,
                                            model.z_cont: z_cont,
                                            model.z_rand: z_rand})

            summary, global_step = sess.run([model.summary_op, model.global_step],
                                    feed_dict={model.x: x_batch,
                                               model.z_cat: z_cat,
                                               model.z_cont: z_cont,
                                               model.z_rand: z_rand})

            if step % 100 == 0:
                print('Epoch[{}/{}] Step[{}/{}] g_loss:{:.4f}, d_loss:{:.4f}'.format(epoch, args.epoch, step,
                                                                                     steps_per_epoch, g_loss,
                                                                                     d_loss))
            summary_writer.add_summary(summary, global_step)
            
        model.save(sess, args.model_dir, global_step)
        
        

def inference(args, model, sess):

        if args.model_dir is None:
            raise ValueError('Need to provide model directory')

        # load model
        model.load(sess, args.model_dir)

        for q in range(args.num_cont):
            col = []
            for c in range(args.num_category):
                row = []
                for d in range(11):
                    z_cat = [c]
                    z_cont = np.zeros([1, args.num_cont])
                    z_cont[0,q] = -2 + d*0.4     # -2 ~ 2
                    z_rand = np.random.uniform(-1, 1, size=[1, args.num_rand])

                    g = sess.run([model.g], feed_dict={model.z_cat: z_cat,
                                                           model.z_cont: z_cont,
                                                           model.z_rand: z_rand})
                    g = np.squeeze(g)
                    multiplier = 255.0 / g.max()
                    g = (g * multiplier).astype(np.uint8)
                    row.append(g)

                row = np.concatenate(row, axis=1)
                col.append(row)
            result = np.concatenate(col, axis=0)
            filename = 'continuous_' + str(q) + '_col_cat_row_change.png'
            cv2.imwrite(os.path.join(model.test_dir, filename), result)
