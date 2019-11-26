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

version='Superslomo_v1_withtime_'

base_lr=0.0001
beta1=0.5
lr_rate = base_lr #tf.train.exponential_decay(base_lr,  global_step=self.global_step, decay_steps=decay_steps, decay_rate=decay_rate)


class Slomo_step2(Slomo_flow): 
    def __init__(self, sess, modelpath=modelpath):
        super().__init__( sess, modelpath)
        self.last_optical_flow=self.graph.get_tensor_by_name("second_last_flow:0")
        self.last_optical_flow_shape=self.last_optical_flow.get_shape().as_list()
        
        try:
            self.out_last_flow=self.graph.get_tensor_by_name("second_unet/strided_slice_89:0")
            self.G_loss_all_for_finetune=self.graph.get_tensor_by_name("G_loss_all_for_finetune:0")
            
            self.train_op_G = tf.train.AdamOptimizer(lr_rate, beta1=beta1, name="superslomo_adam_G_fintune").minimize(self.G_loss_all_for_finetune,  \
                                                                                               var_list=self.first_para+self.sec_para  )
            
            
        except Exception as e:
            print ("loading second_unet/strided_slice_89:0 error, give it to the child")
        
        
    def process_one_video(self, interpola_cnt, inpath, outpath, keep_shape=True, withtrain=False):
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
        kep_last_flow_train=np.zeros(self.last_optical_flow_shape)
        
        success=True
        seri_frames=[]

        cnt=0
        while success:     
            success, frame= videoCapture.read()
            
            if frame is not None:
                if not keep_shape: frame=cv2.resize(frame, self.videoshape)
                seri_frames.append(frame)
                if len(seri_frames)<self.batch+2: continue
            else: success=False
            
            #训练一次
            if withtrain and len(seri_frames)==self.batch+2: 
                _,kep_last_flow_train=self.train_once(seri_frames,  kep_last_flow_train)
            
            if len(seri_frames)<2: continue
            sttime=time.time()              
            #outimgs  [interpolate, len(seri_frames)-1]
            end_frame_ind=min( len(seri_frames), int(self.batch+1) )
            outimgs, kep_last_flow=self.getframes_throw_flow(seri_frames[: end_frame_ind ], interpola_cnt, kep_last_flow)
            #write imgs to video
            for i in range(end_frame_ind-1):
                #print (seri_frames[i].shape)
                videoWrite.write(seri_frames[i])
                for j in range(interpola_cnt):
                    tepimgs=outimgs[j][i]
                    if keep_shape: tepimgs=cv2.resize(tepimgs, size)
                    #print (tepimgs.shape)
                    videoWrite.write( tepimgs )
            
            cnt+=end_frame_ind-1
            print (cnt,'/',frame_cnt,'  time gap:',time.time()-sttime)
            seri_frames=seri_frames[end_frame_ind-1:]
            
            
        if len(seri_frames)>=1: 
            #print (seri_frames[-1].shape)
            videoWrite.write(seri_frames[-1])
        
        videoWrite.release()
        videoCapture.release()
        self.show_video_info( outpath)
        
        outh264path=op.splitext(outpath)[0]+'_h264.mp4'
        self.convert_mp4_h264(outpath, outh264path, False)
        print ("conver to h264 formate done!")
        
        return fps
        '''
        outgifpath=op.splitext(outpath)[0]+'.gif'
        print ('for convent, converting mp4->gif:',outpath,'->',outgifpath)
        self.convert_mp42gif(outpath, outgifpath)
        
        print ("for ppt show,merging two videos:")
        outgifpath=op.splitext(outpath)[0]+'_merged.gif'
        self.merge_two_videos(inpath, outpath, outgifpath)
        '''
        
    def train_once(self, seri_frames, lastflow, timerates=0.5):
        seri_frames_len=len(seri_frames)
        if seri_frames_len<self.batch+2:  return None
        
        placetep=np.zeros(self.placeimgshape)
        st=time.time()
        for j in range( seri_frames_len-2 ):
            placetep[j,:,:,:3]=cv2.resize(seri_frames[j], self.imgshape)
            placetep[j,:,:,3:6]=cv2.resize(seri_frames[j+1], self.imgshape)
            placetep[j,:,:,6:]=cv2.resize(seri_frames[j+2], self.imgshape)
        placetep=self.img2tanh(placetep)
        timerates_extend=np.array(timerates)*np.ones(self.batch, dtype=np.float32)  #这么写可以自动扩展，自适应timerates为sacale或list
        
        G_loss_all_for_finetune, outflow, _=self.sess.run([self.G_loss_all_for_finetune, self.out_last_flow, self.train_op_G], \
                        feed_dict={  self.img_pla:placetep , self.training:True, self.timerates:timerates_extend, self.last_optical_flow:lastflow})
        print (  "train once(one batch), time:%f , loss:%f "%(time.time()-st, G_loss_all_for_finetune  )  ) 
        return G_loss_all_for_finetune, outflow
        
    def getframes_throw_flow(self, seri_frames, interpola_cnt, last_flow):
        '''
        重写父类中的该函数，对应加上time step的网络
        这里是第一种方法获取中间帧，直接获得通过网络G输出的帧而不是光流，但这样帧的大小是固定的
        cnt:中间插入几帧,这里由于
        return :[interpola_cnt, len(seri_frames)-1]个图片，列优先便利是时序
        '''
        timerates=[i*1.0/(interpola_cnt+1) for i in range(1,interpola_cnt+1)]
        placetep=np.zeros(self.placeimgshape)
        
        ret=[]
        st=time.time()
        for i in range(interpola_cnt):
            for j in range( len(seri_frames)-1 ):
                placetep[j,:,:,:3]=cv2.resize(seri_frames[j], self.imgshape)
                placetep[j,:,:,6:]=cv2.resize(seri_frames[j+1], self.imgshape)
                
            placetep=self.img2tanh(placetep)
            outimg, outflow=self.sess.run([self.outimg, self.out_last_flow], feed_dict={  self.img_pla:placetep , self.training:False, self.timerates:[timerates[i]]*self.batch, self.last_optical_flow:last_flow})
            
            ret.append(self.tanh2img(  outimg[:len(seri_frames)-1]   ))
        print ("run %d iters, time:%f"%(interpola_cnt, time.time()-st))
        return ret, outflow

    
    
    def eval_on_one_video(self, interpola_cnt, inpath):
        '''
        inpath:inputvideo's full path
        interpolatecnt:中间拿出来多少帧作为GT与结果进行评估
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
        video_sttime=time.time()
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
        print ("this video timeused:",time.time()-video_sttime)


    def process_on_one_frame_dir(self, interpola_cnt, inpath, outpath):
        '''
        注意这里是进行补帧的过程而不是evaluate，同video一个道理，但是这里是已经将video分解了
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
            print (imgshape,"-->",self.videoshape)
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
                    
                    outimgname="frame%s_%d_%s.png"%(kep_img_names[j], k,  kep_img_names[j+1])
                    outname=op.join(outpath, outimgname)
                    cv2.imwrite( outname  ,tep_image)
                    
                    
            seri_frames=[ seri_frames[-1]  ]
            kep_img_names=[ kep_img_names[-1] ]
        
        
    def generate_middlebury_allframes(self, middleburey_path=middleburey_path, interpolate_cnt=1):
        '''
        针对middlebury数据的benchmark进行生成中间帧的代码，这里数据集中的中间帧需要提交到官网进行评估
        '''
        inputdir=op.join(middleburey_path, "eval-data")
        outdir=op.join(middleburey_path, "output_interframes")
        os.makedirs(outdir,  exist_ok=True)
        
        for i in os.listdir(inputdir):
            tep_in=op.join(inputdir, i)
            tep_out=op.join(outdir, i)
            os.makedirs(tep_out, exist_ok=True)
            self.process_on_one_frame_dir(interpolate_cnt, tep_in, tep_out)

    def eval_on_one_framedir(self, interpola_cnt, inpath, scale=0.5):
        '''
        inpath:inputvideo's full path
        interpolatecnt:中间拿出来多少帧作为GT与结果进行评估
        scale:输入图像分辨率乘上scale是两者共同要达到的分辨率，即图像和生成帧都resize到这个分辨率
        '''
        print ("Evaluating frame dir:",inpath)
        frame_list=os.listdir(inpath)
        frame_list.sort()
        frame_list=[var for var in frame_list if op.splitext(var)[-1].lower() in ['.png','.jpg','.tif']]
        kep_last_flow=np.zeros(self.last_optical_flow_shape)
        seri_frames=[]
        inter_frames=[]
        kep_psnr=[]
        kep_ssim=[]

        for ind,i in enumerate(frame_list): 
            tepdir=op.join(inpath, i)
            tepimg=cv2.imread(tepdir)
            if tepimg is None:print (tepdir)
            if scale>0: #scale大于0则将图片缩放到scale，否者就按照网络输出大小判定
                targetimgshape= (np.array( tepimg.shape)*scale ).astype(np.int)
                targetimgshape=(targetimgshape[1], targetimgshape[0])
            else:
                targetimgshape=self.videoshape
            tepimg=cv2.resize(tepimg, targetimgshape)
            
            if (ind)%(interpola_cnt+1) ==0  : seri_frames.append(tepimg)
            else: inter_frames.append(tepimg)                
                
            if ind<( len(frame_list)-1) and len(seri_frames)<self.batch+1: continue
            
            if len(seri_frames)<2: break
            
            sttime=time.time()              
            
            outimgs, kep_last_flow=self.getframes_throw_flow(seri_frames, interpola_cnt, kep_last_flow)
            #write imgs to video
            #outimgs  [interpolate, len(seri_frames)-1]
            inter_frames_cnt=0
            for i in range(len(seri_frames)-1):
                for j in range(interpola_cnt):
                    psnr=skimage.measure.compare_psnr(inter_frames[inter_frames_cnt], cv2.resize(outimgs[j][i],  targetimgshape ), 255)
                    ssim=skimage.measure.compare_ssim(inter_frames[inter_frames_cnt], cv2.resize(outimgs[j][i],  targetimgshape ), multichannel=True)
                    
                    kep_psnr.append(psnr)
                    kep_ssim.append(ssim)
                    
                    inter_frames_cnt+=1
                    
                    
            print (ind,'/',len(frame_list),'  time gap:',time.time()-sttime)
            seri_frames=[ seri_frames[-1]  ]
            inter_frames=[]        
        print ("mean psnr:", np.mean(kep_psnr))
        print ("mean ssim:", np.mean(kep_ssim))
        
    def eval_on_framdirs(self, interpola_cnt, rootdir, scale=0.5):
        for i in os.listdir(rootdir):
            tepdir=op.join(rootdir, i)
            if op.isdir(tepdir):  
                #print ("evalutating dir:",tepdir)
                self.eval_on_one_framedir(interpola_cnt, tepdir, scale)
                
    def eval_on_ucf_mini(self, ucf_path=ucf_path, outdir="my_frames"):
        '''
        :在ucf的superslomo提供的结果数据上进行生成并对比(lstm version)
        '''
        outframepath=op.join(ucf_path, outdir)
        os.makedirs(outframepath, exist_ok=True)
        
        kep_last_flow=np.zeros(self.last_optical_flow_shape)
        
        inframepath=op.join(ucf_path, "ucf101_interp_ours")
        ind_dirs=os.listdir(inframepath)
        for ind,i in enumerate(ind_dirs):
            tep_in=op.join(inframepath, i)
            tep_out=op.join(outframepath, i)
            os.makedirs(tep_out, exist_ok=True)
            
            frame0_p=op.join(tep_in, "frame_00.png")
            frame2_p=op.join(tep_in, "frame_02.png")
            
            frame1_my_p=op.join(tep_out, "frame_01_my.png")
            
            frame0=cv2.imread(frame0_p)
            frame2=cv2.imread(frame2_p)
            
            seri_frames=[frame0, frame2]
            outimgs, _=self.getframes_throw_flow(seri_frames, 1, kep_last_flow)
            #[interpola_cnt, len(seri_frames)-1]个图片，列优先便利是时序
            outf=outimgs[0][0]
            
            print (ind,"/",len(ind_dirs),outf.shape)
            outf=cv2.resize(outf, (256, 256))
            cv2.imwrite(frame1_my_p, outf)
            

if __name__=='__main__':
    with tf.Session() as sess:
        #slomo=Slomo_flow(sess)
        slomo=Slomo_step2(sess)
        #slomo=Step_two(sess)
        slomo.process_video_list(inputvideo, outputvideodir, 1, version)
        #slomo.eval_video_list(inputvideo,  2)
        #slomo.eval_on_ucf_mini(ucf_path)
       
        
        
        
        
        
    
         
    