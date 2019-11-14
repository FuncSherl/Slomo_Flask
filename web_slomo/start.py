#coding:utf-8
'''
Created on 2019年11月13日

@author: sherl
'''
from flask import Flask
from datetime import datetime,timedelta
import sys,uuid,os,werkzeug
import os.path as op
from flask import Flask,url_for,request,render_template,session


#reload(sys)
#sys.setdefaultencoding("utf-8")

#这里应该将static_url_path设为空，否则html中的每个资源连接都要以static开头才行，但是static_folder不要动，当来一个请求url时，会到static_folder下找静态文件，但是也会匹配static_url_path开头
app = Flask(__name__, static_url_path='')  # ,static_folder='',

def random_filename(filename):
    ext = os.path.splitext(filename)[1]
    new_filename = uuid.uuid4().hex + ext
    return new_filename


#主页面---------------------------------------------------------------------------
@app.route('/',methods=['GET','POST'])#  主页面
def main_page():
    app.logger.debug(session)
    return render_template('index.html', movie_name_ori="home", movie_name_slomo="home")


#----------------------------------------------------------------------------------------
#下面错误处理

@app.errorhandler(404)
def not_found(e):
    print (request.url)
    return '404 not found <h1>'+request.url+'</h1>'

#---------------------------------------------------------------------------------------

if __name__ == '__main__':
    #print (url_for('static',filename='1.jpg') )
    app.debug = True#不可用于发布版本
    app.send_file_max_age_default=timedelta(seconds=1)
    app.config['MAX_CONTENT_LENGTH'] = 30 * 1024 * 1024   #  ...B
    app.run(host='0.0.0.0',port=8000)
    
    
    
    