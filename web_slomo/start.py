#coding:utf-8
'''
Created on 2019年11月13日

@author: sherl
'''
from flask import Flask
from datetime import datetime,timedelta
import sys,uuid,os,werkzeug,shutil,cv2
import os.path as op
from flask import Flask,url_for,request,render_template,session,redirect
import superslomo_lstm_test as slomo_model
import tensorflow as tf



#reload(sys)
#sys.setdefaultencoding("utf-8")

#这里应该将static_url_path设为空，否则html中的每个资源连接都要以static开头才行，但是static_folder不要动，当来一个请求url时，会到static_folder下找静态文件，但是也会匹配static_url_path开头
app = Flask(__name__, static_url_path='')  # ,static_folder='',

upload_path="./static/upload"
video_path="./static/video"

getfilename=lambda x:op.splitext(  op.split(x)[-1]  )[0]

def convert_fps(inpath, outpath, intercnt):
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
    if op.exists(inpath): os.remove(inpath)
    
    return convert_mp4_h264(tep_path, outpath)


def convert_mp4_h264( inpath, outpath):
    cmdstr="ffmpeg -i %s -vcodec libx264 -f mp4 %s"%(inpath, outpath)
    print (cmdstr)
    retn = os.system(cmdstr)
    if not retn:  #right
        if op.exists(inpath): os.remove(inpath)
        return outpath
    if op.exists(outpath): os.remove(outpath)
    return inpath

def clean_videos():
    if op.exists(upload_path): shutil.rmtree(upload_path)
    if op.exists(video_path):  shutil.rmtree(video_path)
    

def random_filename(filename):
    ext = os.path.splitext(filename)[1]
    new_filename = uuid.uuid4().hex + ext
    return new_filename


#主页面---------------------------------------------------------------------------
@app.route('/',methods=['GET','POST'])#  主页面
def main_page():
    app.logger.debug(session)
    
    uploadvideo_path=outpath='home.mp4'
    tep=session.get('file_after')
    if tep is not None:
        outpath=tep
    tep=session.get('file_fps')
    if tep is not None:
        uploadvideo_path=tep
    
        
    return render_template('index.html', movie_name_ori=getfilename( uploadvideo_path), movie_name_slomo=getfilename(outpath)  )

sess=tf.Session()
slomo=slomo_model.Slomo_step2_LSTM(sess)

@app.route('/upload_file',methods=['POST'])#  file
def uploadfile():    
    print (request)
    #删除上一次的video
    tep=session.get('file_after')
    if tep is not None:
        if op.exists(tep): os.remove(tep)
        print ("remove ",tep)
        session.pop("file_after")
    #删除上一次的video
    tep=session.get('file_fps')
    if tep is not None:
        if op.exists(tep): os.remove(tep)
        print ("remove ",tep)
        session.pop("file_fps")
    ###################################################################
    intercnt=request.form.get('intercnt', type=int,default=1)
    files = request.files.get('file')
    
    filename="home.mp4"
    
    if files is not None:
        filename=random_filename(files.filename)
        print (filename)
        
        uploadvideo_path=op.join(upload_path, filename)
        files.save( uploadvideo_path)
        
        outpath=op.join(video_path, filename)

        oldfps=slomo.process_one_video( intercnt, uploadvideo_path, outpath, keep_shape=False, withtrain=False)
            
        #convert to h264
        outpath_h264=op.splitext(outpath)[0]+"_h264.mp4"
            
        outpath=convert_mp4_h264(outpath, outpath_h264)
            
        session['file_after']=outpath
        #convert fps
        uploadvideo_path_fps=op.splitext(uploadvideo_path)[0]+"_fps.mp4"
        uploadvideo_path=convert_fps(uploadvideo_path, uploadvideo_path_fps, intercnt)
        
        session['file_fps']=uploadvideo_path    
        #render_template('index.html', movie_name_ori=getfilename( uploadvideo_path), movie_name_slomo=getfilename(outpath)  )
    return redirect( url_for("main_page") )
    

#----------------------------------------------------------------------------------------
#下面错误处理

@app.errorhandler(404)
def not_found(e):
    print ("not found:",request.url)
    return '404 not found <h1>'+request.url+'</h1>'

#---------------------------------------------------------------------------------------

if __name__ == '__main__':
    clean_videos()
    
    os.makedirs(upload_path,  exist_ok=True)
    os.makedirs(video_path,  exist_ok=True)
    
    #app.debug = True#不可用于发布版本
    app.send_file_max_age_default=timedelta(seconds=1)
    app.config['MAX_CONTENT_LENGTH'] = 20 * 1024 * 1024   #  30M
    app.secret_key = os.urandom(24)
    app.run(host='0.0.0.0',port=8000)
    
    
    
    