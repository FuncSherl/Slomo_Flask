'''
Created on Apr 9, 2019

@author: sherl
'''

import tensorflow as tf
import numpy as np
import os.path as op
#import matplotlib.pyplot as plt
import cv2,os,time
from datetime import datetime
import skimage
import imageio
from superslomo_test_withtime import *


modelpath="Pictures/superslomo/SuperSlomo_2019-11-09_15-57-28_base_lr-0.000100_batchsize-10_maxstep-240000_LSTM_Version_fixshape"
modelpath="Pictures/superslomo/SuperSlomo_2019-11-13_17-28-10_base_lr-0.000100_batchsize-6_maxstep-240000_TrainWith360pVersion"
#modelpath="Documents/xvhao/use_tensorflow/use_tensor/GAN_slomo/logs_superslomo/SuperSlomo_2019-11-13_17-28-10_base_lr-0.000100_batchsize-6_maxstep-240000"
modelpath=op.join(homepath, modelpath)


version='Superslomo_v2_lstm_fixshape_'

class Slomo_step2_LSTM(Slomo_step2): 
    def __init__(self, sess, modelpath=modelpath):
        super().__init__( sess, modelpath)
        #self.last_optical_flow=self.graph.get_tensor_by_name("second_last_flow:0")
        #self.last_optical_flow_shape=self.last_optical_flow.get_shape().as_list()
        
        #self.out_last_flow=self.graph.get_tensor_by_name("second_unet/strided_slice_89:0")
        self.out_last_flow=self.graph.get_tensor_by_name("second_unet/second_batch_last_flow:0")
        
        self.sess.run(tf.global_variables_initializer())
        self.saver.restore(self.sess, tf.train.latest_checkpoint(modelpath))


if __name__=='__main__':
    with tf.Session() as sess:
        #slomo=Slomo_flow(sess)
        slomo=Slomo_step2_LSTM(sess)
        #slomo=Step_two(sess)
        slomo.process_video_list(inputvideo, outputvideodir, 1, version, keep_shape=True,withtraining=True)
        #slomo.eval_video_list(inputvideo,  1)
        #slomo.eval_on_ucf_mini(ucf_path)
        #slomo.generate_middlebury_allframes(middleburey_path, 1)
        #slomo.eval_on_framdirs(31, slowflow_train, -1)  #31 slowflow_train    #1 7 MPI_sintel_clean
        #slomo.eval_on_ucf_mini()
        
        
        
        
        
    
         
    