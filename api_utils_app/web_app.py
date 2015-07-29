from flask import Flask, jsonify, render_template, request, session, url_for
import json
from collections import defaultdict, OrderedDict
from luminoso_api import LuminosoClient

app = Flask(__name__)
app.secret_key = 'secret_key_that_we_need_to_have_to_use_sessions'

@app.route('/')
def home():
    return render_template('login.html')

@app.route('/', methods=['GET', 'POST'])
def login():
	session['username'] = request.form['username']
	session['password'] = request.form['password']
	session['apps_to_show'] = {'app1':url_for('app1')}
	try:
		LuminosoClient.connect(username = session['username'], password = session['password'])
		#return redirect(url_for('index')) this wasn't working...perhaps because since we haven't rendered the template of index before, it's not possible to redirect? 
		return render_template('index.html')
	except:
		error = 'Invalid_credentials'
		return render_template('login.html', error=error)

@app.route('/index')
def index():
    return render_template('index.html', apps=session['apps_to_show'])

@app.route('/app1')
def app1():
	return render_template('lumi_app1.html')

if __name__ == '__main__':
    app.run(debug=True)