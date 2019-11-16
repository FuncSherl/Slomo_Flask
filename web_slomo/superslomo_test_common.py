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


homepath=os.path.expanduser('~')
print (homepath)
#modelpath="Pictures/superslomo/SuperSlomo_2019-11-03_20-16-01_base_lr-0.000100_batchsize-10_maxstep-240000_add_step2_time_sequence"
modelpath=r'/home/sherl/Pictures/superslomo/SuperSlomo_2019-11-02_13-56-35_base_lr-0.000100_batchsize-10_maxstep-240000_original_paper'

modelpath=op.join(homepath, modelpath)

meta_name=r'model_keep-239999.meta'

ucf_path=r'/media/sherl/本地磁盘/data_DL/UCF101_results'
middleburey_path=r"/media/sherl/本地磁盘/data_DL/eval-color-allframes"

version='Superslomo_v1_'
'''
inputvideodir='./testing_gif'
outputvideodir='./outputvideos'   #输出的video的路径，会在该路径下新建文件夹

os.makedirs(inputvideodir,  exist_ok=True)
os.makedirs(outputvideodir,  exist_ok=True)

video_lists=os.listdir(inputvideodir)  #['original.mp4', 'car-turn.mp4']  #
inputvideo = [op.join(inputvideodir, i.strip()) for i in video_lists ]  #这里保存所有需要测的video的fullpath，后面根据这里的list进行测试

'''

mean_dataset=[102.1, 109.9, 110.0]

class Slomo_flow:
    def __init__(self,sess, modelpath=modelpath):
        self.sess=sess
        print ("loading model:",modelpath)
        saver = tf.train.import_meta_graph(op.join(modelpath, meta_name) )
        saver.restore(self.sess, tf.train.latest_checkpoint(modelpath))
        
        # get weights
        self.graph = tf.get_default_graph()
        self.outimg = self.graph.get_tensor_by_name("second_outputimg:0")
        self.optical_t_0=self.graph.get_tensor_by_name("second_opticalflow_t_0:0")
        self.optical_t_2=self.graph.get_tensor_by_name("second_opticalflow_t_1:0")
        self.optical_0_1=self.graph.get_tensor_by_name("first_opticalflow_0_1:0")
        self.optical_1_0=self.graph.get_tensor_by_name("first_opticalflow_1_0:0")
        
        #self.occu_mask=self.graph.get_tensor_by_name("prob_flow1_sigmoid:0")
        
        #placeholders
        self.img_pla= self.graph.get_tensor_by_name('imgs_in:0')
        self.training= self.graph.get_tensor_by_name("training_in:0")
        self.timerates= self.graph.get_tensor_by_name("timerates_in:0")
        #self.last_optical_flow=self.graph.get_tensor_by_name("second_last_flow:0")
        
        print (self.outimg)
        
        self.optical_flow_shape=self.optical_t_0.get_shape().as_list() #[12, 180, 320, 2]
        #print (self.optical_flow_shape)
        self.placeimgshape=self.img_pla.get_shape().as_list() #[12, 180, 320, 9]
        self.batch=self.placeimgshape[0]
        self.imgshape=(self.placeimgshape[2], self.placeimgshape[1]) #w*h
        
        self.outimgshape=self.outimg.get_shape().as_list() #self.outimgshape: [12, 180, 320, 3]
        self.videoshape=(self.outimgshape[2], self.outimgshape[1]) #w*h
        
    def getframes_throw_flow(self, frame0, frame2, cnt):
        '''
        这里是第一种方法获取中间帧，直接获得通过网络G输出的帧而不是光流，但这样帧的大小是固定的
        cnt:中间插入几帧
        '''
        if cnt>self.batch: 
            print ('error:insert frames cnt should <= batchsize:',self.batch)
            return None
            
        timerates=[i*1.0/(cnt+1) for i in range(1,self.batch+1)]
        
        frame0=cv2.resize(frame0, self.imgshape)
        frame2=cv2.resize(frame2, self.imgshape)
        
        placetep=np.zeros(self.placeimgshape)
        for i in range(cnt):
            placetep[i,:,:,:3]=frame0
            placetep[i,:,:,6:]=frame2
        
        placetep=self.img2tanh(placetep)
        out=self.sess.run(self.outimg, feed_dict={  self.img_pla:placetep , self.training:False, self.timerates:timerates})
        return self.tanh2img(out[:cnt])
    
    def getflow_to_frames(self, frame0, frame2, cnt):
        '''
        第二种方式合成帧，这里获取的是中间的光流，先resize光流，然后用warp加上原网络中的时间等一系列合成操作，这样能获得任意大小帧，但是要注意拿到光流后的处理要和原网络一样，否则会有问题
        #这里先resize光流，在合成帧，保持原视频分辨率
        '''
        if cnt>self.batch: 
            print ('error:insert frames cnt should <= batchsize:',self.batch)
            return None
        fshape=frame0.shape
        resize_sha=(fshape[1], fshape[0]) #width,height
        timerates=[i*1.0/(cnt+1) for i in range(1,self.batch+1)]
        placetep=np.zeros(self.placeimgshape)
        for i in range(cnt):
            placetep[i,:,:,:3]=cv2.resize(frame0, self.imgshape)
            placetep[i,:,:,6:]=cv2.resize(frame2, self.imgshape)
        
        placetep=self.img2tanh(placetep)
        
        flowt_0,flowt_2,flow0_2, flow2_0=self.sess.run([self.optical_t_0, self.optical_t_2, self.optical_0_1, self.optical_1_0], feed_dict={  self.img_pla:placetep , self.training:False, self.timerates:timerates})
        #!!!!!!!!!!!!
        return self.frameandflow2frames(frame0, frame2, flow0_2, flow2_0, cnt)
    
    
    def frameandflow2frames(self, frame0, frame2, flow0_2, flow2_0, cnt):
        '''
        由前后帧和光流合成中间的cnt帧
        frame0:一帧，前帧
        frame2:一帧，后帧
        flow0_2/flow2_0:光流
        cnt:中间插入几帧
        
        return:cnt 个中间帧[cnt, h, w, 3]
        '''
        timerates=[i*1.0/(cnt+1) for i in range(1,cnt+1)]
        fshape=frame0.shape
        resize_sha=(fshape[1], fshape[0])
        '''
        x,y=np.meshgrid([0,1,2],[0,1])
        >>> x
        array([[0, 1, 2],
               [0, 1, 2]])
        >>> y
        array([[0, 0, 0],
               [1, 1, 1]])
        '''
        X, Y = np.meshgrid(np.arange(fshape[1]), np.arange(fshape[0]))  #w,h 这里X里面代表列，Y代表行号
        xy=np.array( np.stack([Y,X], -1), dtype=np.float32)
        
        #out[x,y]=src[mapx[x,y], mapy[x,y]] or  map[x,y]
        #print (flowt_0.shape,xy.shape)
        out=[]
        for i in range(cnt):
            #这里应该与训练中保持一致
            flowt_0=-(1-timerates[i])*timerates[i]*flow0_2 +  timerates[i]*timerates[i]*flow2_0
            flowt_2=(1-timerates[i])*(1-timerates[i])*flow0_2 + timerates[i]*(timerates[i]-1)*flow2_0
        
            tep0=xy+cv2.resize(flowt_0[i], resize_sha)
            tep1=xy+cv2.resize(flowt_2[i], resize_sha)
            #occu_resize=cv2.resize(occumask[i], resize_sha)
            
            tep0=tep0.astype(np.float32)
            tep1=tep1.astype(np.float32)
            
            #print (tep0)
            #print (tep0[1,2])
            
            #实验中如果是xy=np.array( np.stack([X，Y], -1), dtype=np.float32)，
            #直接将xy作为map送入remap函数中指定映射，则输出和原图一摸一样的图像，要知道前面xy里的[i,j]处对应的值为[j,i],这是一个cv2里的大坑，即map里的（x，y）分别为宽和高，而不是行列坐标 
            '''
            cv2.remap(src, map1, map2, interpolation[, dst[, borderMode[, borderValue]]]) → ds
            dst(x,y)=src(map1(x,y), map2(x,y))
            '''
            
            tepframe0=cv2.remap(frame0, tep0[:,:,1], tep0[:,:,0],  interpolation=cv2.INTER_LINEAR) #self.img_flow_0_t*tep_prob_flow1
            tepframe1=cv2.remap(frame2, tep1[:,:,1], tep1[:,:,0],  interpolation=cv2.INTER_LINEAR) #(1-tep_prob_flow1)*self.img_flow_2_t
            #print (tepframe0[1,2])
            
            
            #occu_resize=np.expand_dims(occu_resize, -1)
            #occumask=np.tile(occumask, [1,1,1,3])
            #time_rate_tep=timerates[i]*(1-occu_resize)+(1-timerates[i])*occu_resize
            final=(1-timerates[i])*tepframe0  +  timerates[i]*tepframe1
            
            #final=tepframe1
            out.append(final)
        out=np.array(out, dtype=np.uint8)
    
        return out
        
    
    def eval_on_ucf_mini(self, ucf_path, outdir="my_frames"):
        '''
        :在ucf的superslomo提供的结果数据上进行生成并对比
        '''
        outframepath=op.join(ucf_path, outdir)
        os.makedirs(outframepath, exist_ok=True)
        
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
            
            outf=self.getframes_throw_flow(frame0, frame2, 1)[0]
            print (ind,"/",len(ind_dirs),outf.shape)
            outf=cv2.resize(outf, (256, 256))
            cv2.imwrite(frame1_my_p, outf)
        
    
    def process_video_list(self, invideolist, outdir, interpola_cnt=7,version=version,  keep_shape=True):
        '''
        入口函数
        输入一个list包含每个video的完整路径：invideolist
        一个输出ideo的路径
        '''
        TIMESTAMP = "{0:%Y-%m-%d_%H-%M-%S}".format(datetime.now())
        outputdir=op.join(outdir, version+TIMESTAMP)
        os.makedirs(outputdir,  exist_ok=True)
        
        for ind,i in enumerate(invideolist):
            fpath,fname=op.split(i.strip())
            
            fname="Slomo_reshape_"+fname
            if keep_shape: fname="Slomo_origin_"+fname            
            
            outputvideo=op.join( outputdir, fname)
            print ('\nvideo:',ind,"/",len(invideolist),"  ",i,'->', outputvideo)
            self.process_one_video(interpola_cnt, i, outputvideo, keep_shape)
    
    def eval_video_list(self, invideolist,  interpola_cnt=7):
        '''
        入口函数
        输入一个list包含每个video的完整路径：invideolist
        一个输出ideo的路径
        '''
        TIMESTAMP = "{0:%Y-%m-%d_%H-%M-%S}".format(datetime.now())
        
        for ind,i in enumerate(invideolist):
            self.eval_on_one_video(interpola_cnt, i)
            
    def eval_on_one_video(self, interpola_cnt, inpath):
        '''
        inpath:inputvideo's full path
        outpath:output video's full path
        keep_shape:if use direct G's output or calculate with optical flow to resize images
        '''
        videoCapture = cv2.VideoCapture(inpath)  
        
        size = (int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        fps=int (videoCapture.get(cv2.CAP_PROP_FPS) )
        frame_cnt=videoCapture.get(cv2.CAP_PROP_FRAME_COUNT) 
        
        print ('\nvideo:',inpath)
        print ('size:',size, '  fps:',fps,'  frame_cnt:',frame_cnt)
        
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
                frame=cv2.resize(frame, self.videoshape)
                
                if (cnt-1)%(interpola_cnt+1) ==0  : seri_frames.append(frame)
                else: inter_frames.append(frame)
                
                
                if len(seri_frames)<2: continue
            else: success=False
            
            if len(seri_frames)<2: break
            
            sttime=time.time()              
            
            outimgs=self.getframes_throw_flow(seri_frames[0], seri_frames[1], interpola_cnt)
            
            print ("len duibi;", outimgs.shape, len(inter_frames))
            
            #print ('get iner frame shape:',outimgs.shape, outimgs.dtype)
            for ind,i in enumerate(outimgs):      
                #print (i.shape) 
                psnr=skimage.measure.compare_psnr(inter_frames[ind], i, 255)
                ssim=skimage.measure.compare_ssim(inter_frames[ind], i, multichannel=True)
                    
                kep_psnr.append(psnr)
                kep_ssim.append(ssim)
                
            #cv2.imshow('t', tepimg)
            #cv2.waitKey()
            
            seri_frames=[seri_frames[-1] ]
            inter_frames=[]

            print (cnt,'/',frame_cnt,'  time gap:',time.time()-sttime)
                        
        videoCapture.release()
        print ("mean psnr:", np.mean(kep_psnr))
        print ("mean ssim:", np.mean(kep_ssim))
        
    
    def process_one_video(self, interpola_cnt, inpath, outpath, keep_shape=True):
        '''
        inpath:inputvideo's full path
        outpath:output video's full path
        keep_shape:if use direct G's output or calculate with optical flow to resize images
        '''
        videoCapture = cv2.VideoCapture(inpath)  
        
        size = (int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        fps=int (videoCapture.get(cv2.CAP_PROP_FPS) )
        frame_cnt=videoCapture.get(cv2.CAP_PROP_FRAME_COUNT) 
        
        print ('video:',inpath)
        print ('size:',size, '  fps:',fps,'  frame_cnt:',frame_cnt)
        
        if not keep_shape:
            videoWrite = cv2.VideoWriter(outpath, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), int (fps), self.videoshape )
            print ('output video:',outpath,'\nsize:',self.videoshape, '  fps:', fps)
        else:
            videoWrite = cv2.VideoWriter(outpath, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), int (fps), size )
            print ('output video:',outpath,'\nsize:',size, '  fps:', fps)
        
        success, frame0= videoCapture.read()
        if not keep_shape: frame0=cv2.resize(frame0, self.videoshape)
        
        success, frame1= videoCapture.read()
        
        
        cnt=0
        while success and (frame1 is not None):
            if frame0 is not None: videoWrite.write(frame0)
            
            if not keep_shape: frame1=cv2.resize(frame1, self.videoshape)
            
            sttime=time.time()              
            
            if not keep_shape: outimgs=self.getframes_throw_flow(frame0, frame1, interpola_cnt)
            else: outimgs=self.getflow_to_frames(frame0, frame1, interpola_cnt)
            
            #print ('get iner frame shape:',outimgs.shape, outimgs.dtype)
            for i in outimgs:      
                #print (i.shape) 
                videoWrite.write(i)
            #cv2.imshow('t', tepimg)
            #cv2.waitKey()
            
            frame0=frame1
            cnt+=1
            print (cnt,'/',frame_cnt,'  time gap:',time.time()-sttime)
            success, frame1= videoCapture.read()
            
            
        videoWrite.write(frame0)
        
        videoWrite.release()
        videoCapture.release()
        self.show_video_info( outpath)
        
        outgifpath=op.splitext(outpath)[0]+'.gif'
        print ('for convent, converting mp4->gif:',outpath,'->',outgifpath)
        self.convert_mp42gif(outpath, outgifpath)
        
        print ("for ppt show,merging two videos:")
        outgifpath=op.splitext(outpath)[0]+'_merged.gif'
        self.merge_two_videos(inpath, outpath, outgifpath)
        
    
    def convert_mp42gif(self, inmp4, outgif):
        videoCapture = cv2.VideoCapture(inmp4)
        fps=int (videoCapture.get(cv2.CAP_PROP_FPS) )
        frame_cnt=videoCapture.get(cv2.CAP_PROP_FRAME_COUNT) 
        kep_frames=[]
        cnt=0
        success, frame= videoCapture.read()
        while success and (frame is not None):
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  #将bgr格式的opencv读取图片转化为rgb
            kep_frames.append(frame)
            cnt+=1
            print (cnt,'/',frame_cnt)
            success, frame= videoCapture.read()
        videoCapture.release()
        #将帧合成gif
        print ('writing to gif:', outgif)
        imageio.mimsave(outgif, kep_frames, 'GIF', duration = 1.0/fps)
        
    def convert_mp4_h264(self, inpath, outpath, delete=True):
        cmdstr="ffmpeg -i %s -vcodec libx264 -f mp4 %s"%(inpath, outpath)
        print (cmdstr)
        retn = os.system(cmdstr)
        if retn:
            print ("error %d exec:%s"%(retn, cmdstr))
            if op.exists(outpath): os.remove(outpath)
            return inpath
        if delete and op.exists(inpath): os.remove(inpath)
        return outpath
    
    def convert_fps(self,inpath, outpath, fps, delete=True):
        '''
        通过调整帧率来控制播放速度，这种由于fps为整数，可能会不够准确，准确的话应该用convert_fps_insert，通过自定插入帧数将每个帧扩充n倍
        '''
        cmdstr="ffmpeg -r %d -i %s -vcodec libx264 -f mp4 %s"%(int(fps), inpath, outpath)
        print (cmdstr)
        retn = os.system(cmdstr)
        if retn:
            print ("error %d exec:%s"%(retn, cmdstr))
            if op.exists(outpath): os.remove(outpath)
            return inpath
        if delete and op.exists(inpath): os.remove(inpath)
        return outpath
    
    def convert_fps_insert(self, inpath, outpath, intercnt, delete=True):
        '''
        将每个帧（除了最后一个）扩充intercnt倍，用于补帧对比
        '''
        videoCapture1 = cv2.VideoCapture(inpath)
        frame_cnt1=videoCapture1.get(cv2.CAP_PROP_FRAME_COUNT)
        fps1=int (videoCapture1.get(cv2.CAP_PROP_FPS) )
        size1 = (int(videoCapture1.get(cv2.CAP_PROP_FRAME_WIDTH)), int(videoCapture1.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        print ("frame cnt:",frame_cnt1)
        tep_path=op.splitext(inpath)[0]+"_tep.mp4"
        
        videoWrite = cv2.VideoWriter(tep_path, cv2.VideoWriter_fourcc(*'MJPG'), int (fps1),  size1)
        
        target_framecnt=(frame_cnt1-1)*(intercnt+1)+1
        cnt=0
        success, frame= videoCapture1.read()
        while success and (frame is not None) and cnt<target_framecnt:
            for i in range(intercnt+1):
                if cnt<target_framecnt:
                    videoWrite.write(frame)
                    cnt+=1
            success, frame= videoCapture1.read() 
        
        videoCapture1.release()
        videoWrite.release()
        if delete and op.exists(inpath): os.remove(inpath)
        
        return self.convert_mp4_h264(tep_path, outpath, delete)
    
        
    def merge_two_videos(self, video1, video2, outgif):
        videoCapture1 = cv2.VideoCapture(video1)
        fps1=int (videoCapture1.get(cv2.CAP_PROP_FPS) )
        frame_cnt1=videoCapture1.get(cv2.CAP_PROP_FRAME_COUNT)
        size1 = (int(videoCapture1.get(cv2.CAP_PROP_FRAME_WIDTH)), int(videoCapture1.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        
        videoCapture2 = cv2.VideoCapture(video2)
        fps2=int (videoCapture2.get(cv2.CAP_PROP_FPS) )
        frame_cnt2=videoCapture2.get(cv2.CAP_PROP_FRAME_COUNT)
        size2 = (int(videoCapture2.get(cv2.CAP_PROP_FRAME_WIDTH)), int(videoCapture2.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        
        size=[min(size1[0],size2[0]), min(size1[1], size2[1])]
        fps=min(fps1, fps2)
        
        rate=max(frame_cnt2, frame_cnt1)/min(frame_cnt2, frame_cnt1)
        mod1=1
        mod2=1
        if frame_cnt1<frame_cnt2: mod1=rate
        else: mod2=rate
        
        cnt1=0
        cnt2=0
        success1, frame1= videoCapture1.read()
        success2, frame2= videoCapture2.read()
        
        gap=10
        frames_kep=[]
        
        while success1 and success2 and (frame1 is not None) and (frame2 is not None):
            tep=np.zeros([size[1], size[0]*2+gap, 3], dtype=np.uint8)
            
            tep[:, :size[0]]=cv2.cvtColor(cv2.resize(frame1, tuple(size)) , cv2.COLOR_BGR2RGB)
            tep[:, -size[0]:]=cv2.cvtColor(cv2.resize(frame2, tuple(size)), cv2.COLOR_BGR2RGB)
            frames_kep.append(tep)
            cnt1+=1
            cnt2+=1
            if cnt1>=mod1:
                cnt1-=mod1
                success1, frame1= videoCapture1.read()
            if cnt2>=mod2:
                cnt2-=mod2
                success2, frame2= videoCapture2.read()
            
        videoCapture1.release()
        videoCapture2.release()
        #将帧合成gif
        print ('writing to gif:', outgif)
        imageio.mimsave(outgif, frames_kep, 'GIF', duration = 1.0/fps)
        
    
        
    
    def show_video_info(self, inpath):
        videoCapture = cv2.VideoCapture(inpath)
        size = (int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        fps=int (videoCapture.get(cv2.CAP_PROP_FPS) )
        
        frame_cnt=videoCapture.get(cv2.CAP_PROP_FRAME_COUNT) 
        
        print ('video:',inpath)
        print ('size:',size, '  fps:',fps,'  frame_cnt:',frame_cnt)
        
        videoCapture.release()
        return size, fps, frame_cnt
    
    def img2tanh(self,img):
        #img=tf.cast(img,tf.float32)
        #img-=mean_dataset*3
        return img*1.0/255
    
    def tanh2img(self,tanhd):
        tep= (tanhd)*255
        #print ('tep.shape:',tep.shape)  #tep.shape: (180, 320, 9)
        multly=int(tep.shape[-1]/len(mean_dataset))
        #print ('expanding:',multly)
        #tep+=mean_dataset*multly
        return tep.astype(np.uint8)
    
    def flow_bgr(self, flow):
        # Use Hue, Saturation, Value colour model 
        h, w = flow.shape[:2]
        hsv = np.zeros((h, w, 3), np.uint8)
        hsv[..., 1] = 255
        
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        '''
        cv2.imshow("colored flow", bgr)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        '''
        return bgr
    
    def after_process(self, img, kernel_size=10):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        #erosion = cv2.erode(img, kernel)  # 腐蚀
        #dilation = cv2.dilate(img, kernel)  # 膨胀
        '''
        先腐蚀后膨胀叫开运算
        opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)  # 开运算
        closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)  # 闭运算
        膨胀：求局部最大值
        腐蚀：局部最小值(与膨胀相反)
        '''
        
        img = cv2.dilate(img, kernel, iterations=1)
        img = cv2.erode(img, kernel, iterations=1)
        return img



class Step_two(Slomo_flow):      
    def process_video_list(self, invideolist, outdir, interpola_cnt=7):
        TIMESTAMP = "{0:%Y-%m-%d_%H-%M-%S}".format(datetime.now())
        outputdir=op.join(outdir, version+"Using_STEP2_"+TIMESTAMP)
        os.makedirs(outputdir,  exist_ok=True)
        
        for ind,i in enumerate(invideolist):
            fpath,fname=op.split(i.strip())            
            outputvideo=op.join( outputdir, "step2_slomo_"+fname)
            print ('video:',ind,"/",len(invideolist),"  ",i,'->', outputvideo)
            self.process_one_video_with_step2(interpola_cnt, i, outputvideo)
    
    def process_one_video_with_step2(self, interpola_cnt, inpath, outpath):
        '''
        inpath:inputvideo's full path
        outpath:output video's full path
        '''
        #这里先构建待会要用到的二次函数的系数矩阵，想办法处理
        scale=0.5
        self.talor_cnt=2  #二次展开
        self.weight0_1_x=np.random.normal(loc=0.0, scale=scale, size=[self.optical_flow_shape[1], self.optical_flow_shape[2], self.talor_cnt+1]) #[180, 320, 3]
        self.weight0_1_y=np.random.normal(loc=0.0, scale=scale, size=[self.optical_flow_shape[1], self.optical_flow_shape[2], self.talor_cnt+1])
        
        self.weight1_0_x=np.random.normal(loc=0.0, scale=scale, size=[self.optical_flow_shape[1], self.optical_flow_shape[2], self.talor_cnt+1]) #[180, 320, 3]
        self.weight1_0_y=np.random.normal(loc=0.0, scale=scale, size=[self.optical_flow_shape[1], self.optical_flow_shape[2], self.talor_cnt+1])
        
        #处理video
        videoname=op.splitext(  op.split(inpath)[-1]  )[0]
        videoCapture = cv2.VideoCapture(inpath)  
        
        size = (int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        fps=int (videoCapture.get(cv2.CAP_PROP_FPS) )
        frame_cnt=videoCapture.get(cv2.CAP_PROP_FRAME_COUNT) 
        
        print ('video:',inpath)
        print ('size:',size, '  fps:',fps,'  frame_cnt:',frame_cnt)
        
        
        videoWrite = cv2.VideoWriter(outpath, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), int (fps), self.videoshape )
        print ('output video:',outpath,'\nsize:',self.videoshape, '  fps:', fps)
       
        
        success=True
        frame=True
        frame_list=[]
        cnt=0
        while success and frame is not None:
            #if frame is not None: videoWrite.write(frame)
            success, frame= videoCapture.read()
            #frame=cv2.resize(frame, self.videoshape)
            
            if success and frame is not None:
                frame_list.append(cv2.resize(frame, self.videoshape))
                cnt+=1
                if len(frame_list)<=self.batch: continue
            
            if len(frame_list)<2: continue
            
            sttime=time.time()              
            
            #这里要求返回值里面包含原来的帧和新加的帧
            outimgs=self.second_step(frame_list, interpola_cnt, cnt-len(frame_list), frame_cnt)
            #print ('get iner frame shape:',outimgs.shape, outimgs.dtype)
            for i in outimgs:      
                #print (type(i), i.shape,i.dtype) 
                videoWrite.write(i)
                
            print (cnt,'/',frame_cnt,'  time gap:',time.time()-sttime)
            frame_list=[ frame_list[-1] ]
            
            
        if len(frame_list)>=1: videoWrite.write(frame_list[-1])
        
        videoWrite.release()
        videoCapture.release()
        self.show_video_info( outpath)
        
        outgifpath=op.splitext(outpath)[0]+'.gif'
        print ('for convent, converting mp4->gif:',outpath,'->',outgifpath)
        self.convert_mp42gif(outpath, outgifpath)
        
        print ("for ppt show,merging two videos:")
        outgifpath=op.splitext(outpath)[0]+'_merged.gif'
        self.merge_two_videos(inpath, outpath, outgifpath)
        
    
    def second_step(self, frames, inter_cnt, framecnt, videoframes):
        '''
        frames:list of frames,max length is batchsize+1
        inter_cnt:hao many frames to insert
        framecnt:当前进行到第几个frame，是一个时间代表
        '''
        lr=0.02
        
        pairs=len(frames)-1
        timerates=[i*1.0/(inter_cnt+1) for i in range(1,self.batch+1)]
        X, Y = np.meshgrid(np.arange(self.optical_flow_shape[2]), np.arange(self.optical_flow_shape[1]))  #w,h 这里X里面代表列，Y代表行号
        row_col=np.array( np.stack([Y,X], -1), dtype=np.float32)
        
        width=self.optical_flow_shape[2]
        height=self.optical_flow_shape[1]
        
        placetep=np.zeros(self.placeimgshape)
        for i in range(pairs):
            placetep[i,:,:,:3]=cv2.resize(frames[i], self.imgshape)
            placetep[i,:,:,6:]=cv2.resize(frames[i+1], self.imgshape)
        
        placetep=self.img2tanh(placetep)
        flow0_2, flow2_0=self.sess.run([ self.optical_0_1, self.optical_1_0], feed_dict={  self.img_pla:placetep , self.training:False, self.timerates:timerates})
        
        print ('one batch flows 0<->2 get,return flow_s shape:',flow0_2.shape)
        ret=[]
        for i in range(pairs):
            print ("frame cnt:",i,'/',pairs)
            ret.append(frames[i])

            timecnt=(framecnt+i)*1.0/videoframes
            timecnt_next=(framecnt+i+1)*1.0/videoframes
            
            #process 0->2
            tep0_1=row_col-flow0_2[i]
            self.weight0_1_x=cv2.remap(self.weight0_1_x, tep0_1[:,:,1], tep0_1[:,:,0],  interpolation=cv2.INTER_LINEAR)
            self.weight0_1_y=cv2.remap(self.weight0_1_y, tep0_1[:,:,1], tep0_1[:,:,0],  interpolation=cv2.INTER_LINEAR)
            
            #如果优化代表从当前位置出发的变化量呢
            #self.weight0_1_x=self.optimize(self.weight0_1_x, timecnt_next, row_col[:,:,1].astype(np.float32)/width, lr)
            #self.weight0_1_y=self.optimize(self.weight0_1_y, timecnt_next, row_col[:,:,0].astype(np.float32)/height, lr)
            
            #self.weight0_1_x=self.optimize(self.weight0_1_x, timecnt, (row_col[:,:,1]-flow0_2[i,:,:,1]).astype(np.float32)/width, lr)
            #self.weight0_1_y=self.optimize(self.weight0_1_y, timecnt, (row_col[:,:,0]-flow0_2[i,:,:,0]).astype(np.float32)/height, lr)
            
            self.weight0_1_x=self.optimize2(self.weight0_1_x,  flow0_2[i,:,:,1], lr)  #这么写就是基于现有坐标的
            self.weight0_1_y=self.optimize2(self.weight0_1_y,  flow0_2[i,:,:,0], lr)
            
            
            
            
            print ("optimize self.weight0_1_x and self.weight0_1_y done...")
            
            self.weight1_0_x=self.optimize2(self.weight1_0_x,  flow2_0[i,:,:,1], lr)
            self.weight1_0_y=self.optimize2(self.weight1_0_y,  flow2_0[i,:,:,0], lr)
            
            #then 2->0
            tep1_0=row_col-flow2_0[i]
            re_tep0_1=self.re_flow(tep1_0)  #由于2->0的光流方向与进行方向相反，这里需要将逆向flow反过来
            
            #self.weight1_0_x=self.optimize(self.weight1_0_x, timecnt, row_col[:,:,1].astype(np.float32)/width, lr)
            #self.weight1_0_y=self.optimize(self.weight1_0_y, timecnt, row_col[:,:,0].astype(np.float32)/height, lr)
            
            #self.weight1_0_x=self.optimize(self.weight1_0_x, timecnt_next, (row_col[:,:,1]-flow2_0[i,:,:,1]).astype(np.float32)/width, lr)
            #self.weight1_0_y=self.optimize(self.weight1_0_y, timecnt_next, (row_col[:,:,0]-flow2_0[i,:,:,0]).astype(np.float32)/height, lr)
            
            print ("optimize self.weight1_0_x and self.weight1_0_y done...")
            
            for ci in range(inter_cnt):
                print ("begin interpolate:",ci,'/',inter_cnt)
                
                time_this=timerates[ci]*(1.0/videoframes)+timecnt
                
                time_this_flow_t_1=self.get_time_flow2(self.weight1_0_y, self.weight1_0_x, timerates[ci], True)
                time_this_flow_t_0=self.get_time_flow2(self.weight1_0_y, self.weight1_0_x, timerates[ci])
                
                #time_this_flow_1_t=self.re_flow(time_this_flow_t_1).astype(np.float32)
                #time_this_flow_0_t=self.re_flow(time_this_flow_t_0).astype(np.float32)     
                
                time_this_flow_1_t= (time_this_flow_t_1 + row_col).astype(np.float32)
                time_this_flow_0_t= (time_this_flow_t_0 + row_col).astype(np.float32)
                
                #print ("time_this_flow_1_t:",type(time_this_flow_1_t), time_this_flow_1_t.dtype)
                
                tepframe0=cv2.remap(frames[i],   time_this_flow_0_t[:,:,1], time_this_flow_0_t[:,:,0],  interpolation=cv2.INTER_LINEAR)
                tepframe1=cv2.remap(frames[i+1], time_this_flow_1_t[:,:,1], time_this_flow_1_t[:,:,0],  interpolation=cv2.INTER_LINEAR)
                
                final=timerates[ci]*tepframe1 + tepframe0*(1-timerates[ci])
                ret.append(final.astype(np.uint8))
            
            self.weight1_0_x=cv2.remap(self.weight1_0_x, re_tep0_1[:,:,1], re_tep0_1[:,:,0],  interpolation=cv2.INTER_LINEAR)
            self.weight1_0_y=cv2.remap(self.weight1_0_y, re_tep0_1[:,:,1], re_tep0_1[:,:,0],  interpolation=cv2.INTER_LINEAR)
        return ret
            
    def get_time_flow(self, weight_row, weight_col, timecnt):
        mid_tep_row=weight_row[:,:,2]*(timecnt)**2 + weight_row[:,:,1]*(timecnt) + weight_row[:,:,0]
        mid_tep_col=weight_col[:,:,2]*(timecnt)**2 + weight_col[:,:,1]*(timecnt) + weight_col[:,:,0]
        
        mid_tep_row*=self.optical_flow_shape[1]
        mid_tep_col*=self.optical_flow_shape[2]
        
        print ("cal flow:",mid_tep_col[3:4,3:5])
        
        return np.stack([mid_tep_row,mid_tep_col], -1)
        
    def optimize(self, weight, timecnt, label, lr=0.01):
        '''
        weight:[180,320,3]
        label:[180, 320] row_col
        '''
        mid_tep=weight[:,:,2]*(timecnt)**2 + weight[:,:,1]*(timecnt) + weight[:,:,0]
        
        loss=mid_tep-label
        print ("loss:",loss[3:4,3:5])
        print ("first weight:",weight[3:4,3:5])
        
        weight[:,:,2]-=loss*lr*(timecnt)**2 
        weight[:,:,1]-=loss*lr*(timecnt)
        weight[:,:,0]-=loss*lr
        
        print ("second weight:",weight[3:4,3:5])
        
        return weight
    
    def get_time_flow2(self, weight_row, weight_col, timecnt, reverse=False):
        if reverse:
            mid_tep_row=weight_row[:,:,2]*(1- (timecnt)**2) + weight_row[:,:,1]*(1-timecnt) #+ weight_row[:,:,0]
            mid_tep_col=weight_col[:,:,2]*(1- (timecnt)**2) + weight_col[:,:,1]*(1-timecnt) #+ weight_col[:,:,0]
        else:
            mid_tep_row=weight_row[:,:,2]*((timecnt)**2) + weight_row[:,:,1]*(timecnt) + weight_row[:,:,0]
            mid_tep_col=weight_col[:,:,2]*((timecnt)**2) + weight_col[:,:,1]*(timecnt) + weight_col[:,:,0]
        #mid_tep_row*=self.optical_flow_shape[1]
        #mid_tep_col*=self.optical_flow_shape[2]
        
        print ("cal flow:",mid_tep_col[3:4,3:5])
        
        return np.stack([mid_tep_row,mid_tep_col], -1)
    
    def optimize2(self, weight, label, lr=0.01):
        #假设x一直为1，值判断当前下一步
        timecnt=1
        mid_tep=weight[:,:,2]*(timecnt)**2 + weight[:,:,1]*(timecnt) + weight[:,:,0]
        
        loss=mid_tep-label
        print ("loss:",loss[3:4,3:5])
        print ("first weight:",weight[3:4,3:5])
        
        weight[:,:,2]-=loss*lr*(timecnt)**2 
        weight[:,:,1]-=loss*lr*(timecnt)
        weight[:,:,0]-=loss*lr
        
        print ("second weight:",weight[3:4,3:5])
        
        return weight
    
    
    def re_flow(self,  flowout):
        '''
        将0->1的flow，转化为1->0的flow
        flow里面是坐标而不是相对坐标
        '''
        shape=flowout.shape
        out=flowout.copy()
        for i in range(shape[0]):
            for j in range(shape[1]):
                oriind_row=flowout[i][j][0]
                oriind_col=flowout[i][j][1]
                if oriind_row>=0 and oriind_col>=0 and oriind_row<shape[0] and oriind_col<shape[1]:
                    out[int(oriind_row)][int(oriind_col)][0]=i-( oriind_row-int(oriind_row) ) 
                    out[int(oriind_row)][int(oriind_col)][1]=j-( oriind_col-int(oriind_col) ) 
        return out
        

if __name__=='__main__':
    with tf.Session() as sess:
        slomo=Slomo_flow(sess)

        #slomo=Step_two(sess)
        slomo.process_video_list(inputvideo, outputvideodir, 6)
        #slomo.eval_video_list(inputvideo,  2)
        #slomo.eval_on_ucf_mini(ucf_path)
       
        
        
        
        
        
    
         
    