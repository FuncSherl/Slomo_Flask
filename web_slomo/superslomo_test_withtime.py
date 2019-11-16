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
from  superslomo_test_common import *


modelpath="Pictures/superslomo/SuperSlomo_2019-11-03_20-16-01_base_lr-0.000100_batchsize-10_maxstep-240000_add_step2_time_sequence"

modelpath=op.join(homepath, modelpath)

version='Superslomo_v1_'

class Slomo_step2(Slomo_flow): 
    def __init__(self, sess, modelpath=modelpath):
        super().__init__( sess, modelpath)
        self.last_optical_flow=self.graph.get_tensor_by_name("second_last_flow:0")
        self.last_optical_flow_shape=self.last_optical_flow.get_shape().as_list()
        
        try:
            self.out_last_flow=self.graph.get_tensor_by_name("second_unet/strided_slice_89:0")
        except Exception as e:
            print ("loading second_unet/strided_slice_89:0 error, give it to the child")
        
        
    def process_one_video(self, interpola_cnt, inpath, outpath, keep_shape=True):
        '''
        inpath:inputvideo's full path
        outpath:output video's full path
        keep_shape:if true:keep original video's shape  false:resize to net shape
        '''
        videoCapture = cv2.VideoCapture(inpath)  
        
        size = (int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        fps=int (videoCapture.get(cv2.CAP_PROP_FPS) )
        frame_cnt=videoCapture.get(cv2.CAP_PROP_FRAME_COUNT) 
        
        print ('video:',inpath)
        print ('size:',size, '  fps:',fps,'  frame_cnt:',frame_cnt)
        

        videoWrite = cv2.VideoWriter(outpath, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), int (fps), (self.videoshape if not keep_shape else size) )
        print ('output video:',outpath)
        print ("keep shape:",keep_shape)
        if keep_shape: print ('size:',size, '  fps:', fps)
        else : print ('size:',self.videoshape, '  fps:', fps)
        
        kep_last_flow=np.zeros(self.last_optical_flow_shape)
        
        success=True
        seri_frames=[]

        cnt=0
        while success:     
            success, frame= videoCapture.read()
            
            if frame is not None:
                if not keep_shape: frame=cv2.resize(frame, self.videoshape)
                seri_frames.append(frame)
                if len(seri_frames)<self.batch+1: continue
            else: success=False
            
            if len(seri_frames)<2: continue
            sttime=time.time()              
            #outimgs  [interpolate, len(seri_frames)-1]
            outimgs, kep_last_flow=self.getframes_throw_flow(seri_frames, interpola_cnt, kep_last_flow)
            #write imgs to video
            for i in range(len(seri_frames)-1):
                #print (seri_frames[i].shape)
                videoWrite.write(seri_frames[i])
                for j in range(interpola_cnt):
                    tepimgs=outimgs[j][i]
                    if keep_shape: tepimgs=cv2.resize(tepimgs, size)
                    #print (tepimgs.shape)
                    videoWrite.write( tepimgs )
            
            cnt+=len(seri_frames)-1
            print (cnt,'/',frame_cnt,'  time gap:',time.time()-sttime)
            seri_frames=[ seri_frames[-1]  ]
            
            
        if len(seri_frames)>=1: 
            #print (seri_frames[-1].shape)
            videoWrite.write(seri_frames[-1])
        
        videoWrite.release()
        videoCapture.release()
        self.show_video_info( outpath)
        return fps
        '''
        outgifpath=op.splitext(outpath)[0]+'.gif'
        print ('for convent, converting mp4->gif:',outpath,'->',outgifpath)
        self.convert_mp42gif(outpath, outgifpath)
        
        print ("for ppt show,merging two videos:")
        outgifpath=op.splitext(outpath)[0]+'_merged.gif'
        self.merge_two_videos(inpath, outpath, outgifpath)
        '''
        
    def getframes_throw_flow(self, seri_frames, interpola_cnt, last_flow):
        '''
        重写父类中的该函数，对应加上time step的网络
        这里是第一种方法获取中间帧，直接获得通过网络G输出的帧而不是光流，但这样帧的大小是固定的
        cnt:中间插入几帧,这里由于
        '''
        timerates=[i*1.0/(interpola_cnt+1) for i in range(1,interpola_cnt+1)]
        placetep=np.zeros(self.placeimgshape)
        
        ret=[]
        
        for i in range(interpola_cnt):
            for j in range( len(seri_frames)-1 ):
                placetep[j,:,:,:3]=cv2.resize(seri_frames[j], self.imgshape)
                placetep[j,:,:,6:]=cv2.resize(seri_frames[j+1], self.imgshape)
                
            placetep=self.img2tanh(placetep)
            outimg, outflow=self.sess.run([self.outimg, self.out_last_flow], feed_dict={  self.img_pla:placetep , self.training:False, self.timerates:[timerates[i]]*self.batch, self.last_optical_flow:last_flow})
            
            ret.append(self.tanh2img(  outimg[:len(seri_frames)-1]   ))
            
        return ret, outflow

    
    
    def eval_on_one_video(self, interpola_cnt, inpath):
        '''
        inpath:inputvideo's full path
        outpath:output video's full path
        #keep_shape:if use direct G's output or calculate with optical flow to resize images
        '''
        videoCapture = cv2.VideoCapture(inpath)  
        
        size = (int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        fps=int (videoCapture.get(cv2.CAP_PROP_FPS) )
        frame_cnt=videoCapture.get(cv2.CAP_PROP_FRAME_COUNT) 
        
        print ('\nvideo:',inpath)
        print ('size:',size, '  fps:',fps,'  frame_cnt:',frame_cnt)
        
        kep_last_flow=np.zeros(self.last_optical_flow_shape)
        
        success=True
        seri_frames=[]
        inter_frames=[]
        kep_psnr=[]
        kep_ssim=[]
        
        cnt=0
        while success:     
            success, frame= videoCapture.read()
            cnt+=1
            
            if frame is not None:
                #frame=cv2.resize(frame, self.videoshape)                
                if (cnt-1)%(interpola_cnt+1) ==0  : seri_frames.append(frame)
                else: inter_frames.append(frame)                
                
                if len(seri_frames)<self.batch+1: continue
            else: success=False
            
            if len(seri_frames)<2: break
            
            sttime=time.time()              
            
            outimgs, kep_last_flow=self.getframes_throw_flow(seri_frames, interpola_cnt, kep_last_flow)
            #write imgs to video
            #outimgs  [interpolate, len(seri_frames)-1]
            inter_frames_cnt=0
            for i in range(len(seri_frames)-1):
                for j in range(interpola_cnt):
                    psnr=skimage.measure.compare_psnr(inter_frames[inter_frames_cnt], cv2.resize(outimgs[j][i],  size ), 255)
                    ssim=skimage.measure.compare_ssim(inter_frames[inter_frames_cnt], cv2.resize(outimgs[j][i],  size ), multichannel=True)
                    
                    kep_psnr.append(psnr)
                    kep_ssim.append(ssim)
                    
                    inter_frames_cnt+=1
                    
                    
            print (cnt,'/',frame_cnt,'  time gap:',time.time()-sttime)
            seri_frames=[ seri_frames[-1]  ]
            inter_frames=[]
            
        videoCapture.release()
        
        print ("mean psnr:", np.mean(kep_psnr))
        print ("mean ssim:", np.mean(kep_ssim))


    def eval_on_one_frame_dir(self, interpola_cnt, inpath, outpath):
        '''
        :处理一个分解后的帧的目录，同video一个道理，但是这里是已经将video分解了
        '''
        print (inpath,"-->",outpath)
        frame_list=os.listdir(inpath)
        frame_list.sort()
        kep_last_flow=np.zeros(self.last_optical_flow_shape)
        seri_frames=[]
        kep_img_names=[]
        cnt=0
        
        for ind,i in enumerate(frame_list):
            tepdir=op.join(inpath, i)
            tepimg=cv2.imread(tepdir)
            imgshape=tepimg.shape
            seri_frames.append(cv2.resize(tepimg, self.videoshape))
            kep_img_names.append( op.splitext(i)[0][5:] )
            cnt+=1
            
            if ind<( len(frame_list)-1) and len(seri_frames)<self.batch+1: continue
            if len(seri_frames)<=1: break
            
            sttime=time.time()              
            
            outimgs, kep_last_flow=self.getframes_throw_flow(seri_frames, interpola_cnt, kep_last_flow)
            print (cnt,'/',len(frame_list),'  time gap:',time.time()-sttime)
            
            
            
            for j in range(len(seri_frames)-1):
                for k in range(interpola_cnt):
                    tep_image=cv2.resize(outimgs[k][j], (imgshape[1], imgshape[0]) )
                    
                    outimgname="frame%si%s.png"%(kep_img_names[j],  kep_img_names[j+1])
                    outname=op.join(outpath, outimgname)
                    cv2.imwrite( outname  ,tep_image)
                    
                    
            seri_frames=[ seri_frames[-1]  ]
            kep_img_names=[ kep_img_names[-1] ]
        
        
    def eval_on_middlebury_allframes(self, middleburey_path, interpolate_cnt=1):
        inputdir=op.join(middleburey_path, "eval-data")
        outdir=op.join(middleburey_path, "output_interframes")
        os.makedirs(outdir,  exist_ok=True)
        
        for i in os.listdir(inputdir):
            tep_in=op.join(inputdir, i)
            tep_out=op.join(outdir, i)
            os.makedirs(tep_out, exist_ok=True)
            self.eval_on_one_frame_dir(interpolate_cnt, tep_in, tep_out)

        

if __name__=='__main__':
    with tf.Session() as sess:
        #slomo=Slomo_flow(sess)
        slomo=Slomo_step2(sess)
        #slomo=Step_two(sess)
        #slomo.process_video_list(inputvideo, outputvideodir, 6, version)
        slomo.eval_video_list(inputvideo,  2)
        #slomo.eval_on_ucf_mini(ucf_path)
       
        
        
        
        
        
    
         
    