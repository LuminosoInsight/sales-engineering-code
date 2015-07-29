from flask import Flask, jsonify, render_template, request, session, url_for
import json
from collections import defaultdict, OrderedDict
from luminoso_api import LuminosoClient
from topic_utilities import copy_topics, del_topics

app = Flask(__name__)
app.secret_key = 'secret_key_that_we_need_to_have_to_use_sessions'

@app.route('/')
def home():
    return render_template('login.html')

@app.route('/', methods=['GET', 'POST'])
def login():
	session['username'] = request.form['username']
	session['password'] = request.form['password']
	session['apps_to_show'] = [('Topic Utilities',url_for('topic_utils')),
							   ('app2',url_for('app2')),
							   ('app3',url_for('app3'))]
	try:
		LuminosoClient.connect('/projects/', username=session['username'],
											 password=session['password'])
		return render_template('welcome.html', urls=session['apps_to_show'])
	except:
		error = 'Invalid_credentials'
		return render_template('login.html', error=error)

@app.route('/index')
def index():
    return render_template('index.html', urls=session['apps_to_show'])

@app.route('/topic_utils')
def topic_utils():
	return render_template('topic_utils.html', urls=session['apps_to_show'])

@app.route('/topic_utils/copy', methods=['POST'])
def topic_utils_copy():
	#NOTE: Should add a checkbox for if the existing topics should be deleted first
	acct = request.form['account']
	source = request.form['source_pid']
	dests = [p_id.strip() for p_id in request.form['dest_pids'].split(',')]
	cli = LuminosoClient.connect('/projects/', username=session['username'],
									   password=session['password'])
	for dest_proj in dests:
		copy_topics(cli, from_acct=acct, from_proj=source, to_acct=acct, to_proj=dest_proj)
	#NOTE: ADD A FLASH CONFIRMATION MESSAGE HERE
	return render_template('topic_utils.html', urls=session['apps_to_show'])


@app.route('/topic_utils/delete', methods=['POST'])
def topic_utils_delete():
	acct = request.form['account']
	dests = [p_id.strip() for p_id in request.form['pids'].split(',')]
	print(acct)
	print(dests)
	cli = LuminosoClient.connect('/projects/', username=session['username'],
									   password=session['password'])
	for dest_proj in dests:
		del_topics(cli, acct_id=acct, proj_id=dest_proj)
	#NOTE: ADD A FLASH CONFIRMATION MESSAGE HERE
	return render_template('topic_utils.html', urls=session['apps_to_show'])

@app.route('/app2')
def app2():
	return render_template('lumi_app2.html')

@app.route('/app3')
def app3():
	return render_template('lumi_app3.html')

if __name__ == '__main__':
    app.run(debug=True)















