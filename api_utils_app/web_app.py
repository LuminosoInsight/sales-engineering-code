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
	session['apps_to_show'] = [('Topic Utilities',url_for('topic_utils')), ('app2',url_for('app2')), ('app3',url_for('app3'))]
	try:
		LuminosoClient.connect(username=session['username'], password=session['password'])
		return render_template('index.html', urls=session['apps_to_show'])
	except:
		error = 'Invalid_credentials'
		return render_template('login.html', error=error)

@app.route('/index')
def index():
    return render_template('index.html', apps=session['apps_to_show'])

@app.route('/topic_utils')
def topic_utils():
	return render_template('topic_utils.html')

@app.route('/topic_utils/copy')
def copy_topics():
	source = request.args.get('source_proj', 0, type=str)
	dests = request.args.get('source_proj', 0, type=str)

@app.route('/topic_utils/delete')
def delete_topics():
	pass

@app.route('/app2')
def app2():
	return render_template('lumi_app2.html')

@app.route('/app3')
def app3():
	return render_template('lumi_app3.html')

if __name__ == '__main__':
    app.run(debug=True)