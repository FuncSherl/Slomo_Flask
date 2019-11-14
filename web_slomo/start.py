#coding:utf-8
'''
Created on 2019年11月13日

@author: sherl
'''
from flask import Flask
from datetime import datetime,timedelta
import sys,uuid,os,werkzeug,shutil
import os.path as op
from flask import Flask,url_for,request,render_template,session
import superslomo_lstm_test as slomo_model
import tensorflow as tf


#reload(sys)
#sys.setdefaultencoding("utf-8")

#这里应该将static_url_path设为空，否则html中的每个资源连接都要以static开头才行，但是static_folder不要动，当来一个请求url时，会到static_folder下找静态文件，但是也会匹配static_url_path开头
app = Flask(__name__, static_url_path='')  # ,static_folder='',

upload_path="./static/upload"
video_path="./static/video"



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
    return render_template('index.html', movie_name_ori="home", movie_name_slomo="home")


@app.route('/upload_file',methods=['POST'])#  file
def uploadfile():
    print (request)
    intercnt=request.form.get('intercnt', type=int,default=1)
    files = request.files.get('file')
    
    if files is not None:
        filename=random_filename(files.filename)
        print (filename)
        file_after=filename
        
        session['filename']=filename
        uploadvideo_path=op.join(upload_path, filename)
        files.save( uploadvideo_path)
        
        outpath=op.join(video_path, filename)
        with tf.Session() as sess:
            #slomo=Slomo_flow(sess)
            slomo=slomo_model.Slomo_step2(sess)
            #slomo=Step_two(sess)
            outpath_h264=slomo.process_one_video( intercnt, uploadvideo_path, outpath)
            os.remove(outpath)
            
            file_after=op.split(  outpath_h264 )[-1]
    
    return render_template('index.html', movie_name_ori=op.splitext(filename)[0], movie_name_slomo=op.splitext(file_after)[0])
    

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
    
    app.debug = True#不可用于发布版本
    app.send_file_max_age_default=timedelta(seconds=1)
    app.config['MAX_CONTENT_LENGTH'] = 20 * 1024 * 1024   #  30M
    app.secret_key = os.urandom(24)
    app.run(host='0.0.0.0',port=8000)
    
    
    
    