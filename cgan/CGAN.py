#-*- coding: utf-8 -*-
from __future__ import division
import os
import time
import tensorflow as tf
import numpy as np

from ops import *
from utils import *
from covers import covers
#import get_images
import random
from matplotlib import pyplot as plt


class CGAN(object):
    model_name = "CGAN" # name for checkpoint

    def __init__(self, sess, epoch, batch_size, z_dim, dataset_name, checkpoint_dir, result_dir, log_dir):
        self.sess = sess
        self.dataset_name = dataset_name
        self.checkpoint_dir = checkpoint_dir
        self.result_dir = result_dir
        self.log_dir = log_dir
        self.epoch = epoch
        self.batch_size = batch_size
        
        if dataset_name == 'mnist' or dataset_name == 'fashion-mnist':
            # parameters
            self.input_height = 100
            self.input_width = 100
            self.output_height = 100
            self.output_width = 100
            self.clip_D = 0
            self.z_dim = z_dim # dimension of noise-vector
            self.y_dim = 19    # dimension of condition-vector (label)
            self.c_dim = 3
            
            # train
            self.learning_rate = 0.0001
            self.beta1 = 0.5
            
            # test
            self.sample_num = 64 # number of generated images to be saved
            
            self.data_X = []
            self.data_y = []
            
            self.num_batches = 32
        else:
            raise NotImplementedError

    def discriminator(self, x, y, is_training=True, reuse=False):

        with tf.variable_scope("discriminator", reuse=reuse):

            # merge image and label
            y = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
            x = conv_cond_concat(x, y)
            
            net = lrelu(conv2d(x, 100, 4, 4, 2, 2, name='d_conv1'))
            net = lrelu(bn(conv2d(net, 128, 4, 4, 2, 2, name='d_conv2'), is_training=is_training, scope='d_bn2'))
            net = tf.reshape(net, [self.batch_size, -1])
            net = lrelu(bn(linear(net, 1024, scope='d_fc3'), is_training=is_training, scope='d_bn3'))
            out_logit = linear(net, 1, scope='d_fc4')
            out = tf.nn.sigmoid(out_logit)
            
            return out, out_logit, net
            
    def generator(self, z, y, is_training=True, reuse=False):

        with tf.variable_scope("generator", reuse=reuse):
            # merge noise and label
            z = concat([z, y], 1)
            
            net = tf.nn.relu(bn(linear(z, 1024, scope='g_fc1'), is_training=is_training, scope='g_bn1'))
            net = tf.nn.relu(bn(linear(net, 128 * 25 * 25, scope='g_fc2'), is_training=is_training, scope='g_bn2'))
            net = tf.reshape(net, [self.batch_size, 25, 25, 128])
            net = tf.nn.relu(bn(deconv2d(net, [self.batch_size, 50, 50, 64], 4, 4, 2, 2, name='g_dc3'),
                                is_training=is_training,scope='g_bn3'))

            out = tf.nn.sigmoid(deconv2d(net, [self.batch_size, 100, 100, 3], 4, 4, 2, 2, name='g_dc4'))
            
            return out

    def build_model(self):
        # some parameters
        image_dims = [self.input_height, self.input_width, self.c_dim]
        bs = self.batch_size
        
        """ Graph Input """
        # images
        self.inputs = tf.placeholder(tf.float32, [bs] + image_dims, name='real_images')
        
        # labels
        self.y = tf.placeholder(tf.float32, [32, self.y_dim], name='y')
        
        # noises
        self.z = tf.placeholder(tf.float32, [32, self.z_dim], name='z')
        
        """ Loss Function """
        
        # output of D for real images
        D_real, D_real_logits, _ = self.discriminator(self.inputs, self.y, is_training=True, reuse=False)
        
        # output of D for fake images
        G = self.generator(self.z, self.y, is_training=True, reuse=False)
        D_fake, D_fake_logits, _ = self.discriminator(G, self.y, is_training=True, reuse=True)
        
        # Sigmoid loss function for discriminator
        #d_loss_real = tf.reduce_mean(
        #				tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real_logits, labels=tf.ones_like(D_real)))
        #d_loss_fake = tf.reduce_mean(
        #				tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.zeros_like(D_fake)))

        #gradient penalty for wasserstein loss
        eps = tf.random_uniform([32, 100, 100, 3], minval = 0., maxval=1.)
        X_inter = eps*self.inputs + (1. - eps)*G
        grad = tf.gradients(self.discriminator(X_inter, self.y, is_training=False, reuse=tf.AUTO_REUSE),[X_inter])[0]
        grad_norm = tf.sqrt(tf.reduce_sum((grad)**2, axis=1))
        grad_pen = 10 * tf.reduce_mean((grad_norm-1)**2)

        #Wasserstein loss for discriminatoy
        d_loss_real = tf.reduce_mean(D_real_logits)
        d_loss_fake = tf.reduce_mean(D_fake_logits)
        self.d_loss = d_loss_real - d_loss_fake + grad_pen
        
        
        # Sigmoid loss function for generator
        #self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.ones_like(D_fake)))
        
        #Wasserstein loss for generator
        self.g_loss = tf.reduce_mean(D_fake_logits)
        
                
        """ Training """
        # divide trainable variables into a group for D and a group for G
        t_vars = tf.trainable_variables()
        d_vars = [var for var in t_vars if 'd_' in var.name]
        g_vars = [var for var in t_vars if 'g_' in var.name]
        

        # optimizers
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.d_optim = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1, beta2 = 0.9) \
                                   .minimize(self.d_loss, var_list=d_vars)
            self.g_optim = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1, beta2 = 0.9) \
                                   .minimize(self.g_loss, var_list=g_vars)
            
            
        """" Testing """
        # for test
        self.fake_images = self.generator(self.z, self.y, is_training=False, reuse=True)
        
        """ Summary """
        d_loss_real_sum = tf.summary.scalar("d_loss_real", d_loss_real)
        d_loss_fake_sum = tf.summary.scalar("d_loss_fake", d_loss_fake)
        d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)
        g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)
        
        # final summary operations
        self.g_sum = tf.summary.merge([d_loss_fake_sum, g_loss_sum])
        self.d_sum = tf.summary.merge([d_loss_real_sum, d_loss_sum])


        
    def train(self):

                                    
        # initialize all variables
        tf.global_variables_initializer().run()
        
        # graph inputs for visualize training results
        self.sample_z = np.random.uniform(-1, 1, size=(self.batch_size , self.z_dim))

        self.test_labels = self.data_y[0:32]
        
        # saver to save model
        self.saver = tf.train.Saver()
        
        # summary writer
        self.writer = tf.summary.FileWriter(self.log_dir + '/' + self.model_name, self.sess.graph)
           
        # restore check-point if it exits
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            start_epoch = (int)(checkpoint_counter / self.num_batches)
            start_batch_id = checkpoint_counter - start_epoch * self.num_batches
            counter = checkpoint_counter
            print(" [*] Load SUCCESS")
        else:
            start_epoch = 0
            start_batch_id = 0
            counter = 1
            print(" [!] Load failed...")

            

        COVERS_PATH = './genres/'
        cover_set = covers(COVERS_PATH, self.batch_size, self.y_dim)
        
        start_time = time.time()
        c_new = 0
        
        for epoch in range(start_epoch, self.epoch):
            self.data_X = []
            self.data_y = []

            for next_idx, batch , batch_cat in cover_set.batched_images(random.randint(0,1550)):
                self.data_X.clear()
                self.data_y.clear()
                self.data_X.append(batch)
                self.data_y.append(batch_cat)
                # get batch data
                for idx in range(1):
                    batch_images = self.data_X[0]
                    batch_labels = self.data_y[0]
                    self.test_labels = batch_labels
                    batch_z = np.random.uniform(-1, 1, [self.batch_size, self.z_dim]).astype(np.float32)
                    
                    # update D network
                    if True: #check == 0:
                        _, summary_str, d_loss = self.sess.run([self.d_optim, self.d_sum, self.d_loss], feed_dict={self.inputs: batch_images, self.y: batch_labels, self.z: batch_z})
                        self.writer.add_summary(summary_str, counter)
                        check = 1
                    else:
                        check = 0
                        
                    if np.mod(epoch, 3) == 0:
                        # update G network
                        _, summary_str, g_loss = self.sess.run([self.g_optim, self.g_sum, self.g_loss],feed_dict={self.y: batch_labels, self.z: batch_z})
                        self.writer.add_summary(summary_str, counter)
                        
                        # display training status
                        print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
                              % (c_new, counter, 2, time.time() - start_time, d_loss, g_loss))
                        c_new += 1
                        
                    # save training results for every 500 steps
                    if np.mod(counter, 500) == 0:
                        samples = self.sess.run(self.fake_images, feed_dict={self.z: self.sample_z, self.y: self.test_labels})
                        tot_num_samples = min(self.sample_num, self.batch_size)
                        manifold_h = int(np.floor(np.sqrt(tot_num_samples)))
                        manifold_w = int(np.floor(np.sqrt(tot_num_samples)))
                        save_images(samples[:manifold_h * manifold_w, :, :, :], [manifold_h, manifold_w],'/' + check_folder(self.result_dir + '/' + self.model_dir) + '/' + self.model_name + '_train_{:02d}_{:04d}.png'.format(epoch, idx))
                        counter += 1
                # After an epoch, start_batch_id is set to zero
                # non-zero value is only for the first epoch after loading pre-trained model
                start_batch_id = 0
                
                if np.mod(epoch, 200) == 0: # save model
                    self.save(self.checkpoint_dir, counter)
                    
                    # show temporal results
                    self.visualize_results(epoch)
                    
                    # save model for final step
            self.save(self.checkpoint_dir, counter)
            
    def visualize_results(self, epoch):
        tot_num_samples = min(self.sample_num, self.batch_size)
        image_frame_dim = int(np.floor(np.sqrt(tot_num_samples)))
        
        """ random condition, random noise """
        y = np.random.choice(self.y_dim, self.batch_size)
        y_one_hot = np.zeros((self.batch_size, self.y_dim))
        y_one_hot[np.arange(self.batch_size), y] = 1
        
        z_sample = np.random.uniform(-1, 1, size=(self.batch_size, self.z_dim))
        
        samples = self.sess.run(self.fake_images, feed_dict={self.z: z_sample, self.y: y_one_hot})
        
        save_images(samples[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim], check_folder(self.result_dir + '/' + self.model_dir) + '/' + self.model_name + '_epoch%03d' % epoch + '_test_all_classes.png')
        
        """ specified condition, random noise """
        n_styles = 19  # must be less than or equal to self.batch_size
        
        np.random.seed()
        si = np.random.choice(self.batch_size, n_styles)
        
        for l in range(self.y_dim):
            y = np.zeros(self.batch_size, dtype=np.int64) + l
            y_one_hot = np.zeros((self.batch_size, self.y_dim))
            y_one_hot[np.arange(self.batch_size), y] = 1
            
            samples = self.sess.run(self.fake_images, feed_dict={self.z: z_sample, self.y: y_one_hot})
            save_images(samples[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim], check_folder(self.result_dir + '/' + self.model_dir) + '/' + self.model_name + '_epoch%03d' % epoch + '_test_class_%d.png' % l)
            
            
            samples = samples[si, :, :, :]
            
            if l == 0:
                all_samples = samples
            else:
                all_samples = np.concatenate((all_samples, samples), axis=0)
                
        """ save merged images to check style-consistency """
        canvas = np.zeros_like(all_samples)
        for s in range(n_styles):
            for c in range(self.y_dim):
                canvas[s * self.y_dim + c, :, :, :] = all_samples[c * n_styles + s, :, :, :]
                
                save_images(canvas, [n_styles, self.y_dim], check_folder(self.result_dir + '/' + self.model_dir) + '/' + self.model_name + '_epoch%03d' % epoch + '_test_all_classes_style_by_style.png')

    @property
    def model_dir(self):
        return "{}_{}_{}_{}".format(self.model_name, self.dataset_name, self.batch_size, self.z_dim)
        
    def save(self, checkpoint_dir, step):
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir, self.model_name)
        
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
            
        self.saver.save(self.sess,os.path.join(checkpoint_dir, self.model_name+'.model'), global_step=step)
            
    def load(self, checkpoint_dir):
        import re
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir, self.model_name)
        
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0
    
