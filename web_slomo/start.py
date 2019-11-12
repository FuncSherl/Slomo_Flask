#coding:utf-8
'''
Created on 2019年11月13日

@author: sherl
'''
from flask import Flask
from datetime import datetime
import sys
from flask import Flask,url_for,request,render_template,session


#reload(sys)
#sys.setdefaultencoding("utf-8")

#这里应该将static_url_path设为空，否则html中的每个资源连接都要以static开头才行，但是static_folder不要动，当来一个请求url时，会到static_folder下找静态文件，但是也会匹配static_url_path开头
app = Flask(__name__, static_url_path='')  # ,static_folder='',



#t
@app.route('/test',methods=['GET','POST'])#request.method == 'POST'  主页面
def deal_test():
    return render_template('usercenter/uc_basic.html')



#主页面---------------------------------------------------------------------------
@app.route('/',methods=['GET','POST'])#  主页面
def main_page():
    
    app.logger.debug(session)
    return render_template('index.html')




#下面是中部导航栏----------------------------------------------------------
@app.route('/little-children',methods=['GET','POST'])#儿童滑雪
def deal_little_child():
    return render_template('little-children.html')






#下面是新闻页面跳转----------------------------------------------------------
@app.route('/natalie',methods=['GET','POST'])#滑雪等级
def deal_natalie():
    return render_template('/news/natalie.html')





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
    app.run(host='0.0.0.0',port=8000)
    
    
    
    