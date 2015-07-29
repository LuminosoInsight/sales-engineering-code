from flask import Flask, jsonify, render_template, request, session, url_for
import json
from collections import defaultdict, OrderedDict
from luminoso_api import LuminosoClient
from topic_utilities import copy_topics, del_topics
from term_utilities import search_terms, ignore_terms, merge_terms

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
							   ('Term Utilities',url_for('term_utils')),
							   ('app3',url_for('topic_utils'))]
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
	acct = request.form['account'].strip()
	source = request.form['source_pid'].strip()
	dests = [p_id.strip() for p_id in request.form['dest_pids'].split(',')]
	cli = LuminosoClient.connect('/projects/', username=session['username'],
											password=session['password'])
	for dest_proj in dests:
		copy_topics(cli, from_acct=acct, from_proj=source, to_acct=acct, to_proj=dest_proj)
	#NOTE: ADD A FLASH CONFIRMATION MESSAGE HERE
	return render_template('topic_utils.html', urls=session['apps_to_show'])


@app.route('/topic_utils/delete', methods=['POST'])
def topic_utils_delete():
	acct = request.form['account'].strip()
	dests = [p_id.strip() for p_id in request.form['pids'].split(',')]
	cli = LuminosoClient.connect('/projects/', username=session['username'],
									   		password=session['password'])
	for dest_proj in dests:
		del_topics(cli, acct_id=acct, proj_id=dest_proj)
	#NOTE: ADD A FLASH CONFIRMATION MESSAGE HERE
	return render_template('topic_utils.html', urls=session['apps_to_show'])

@app.route('/term_utils')
def term_utils():
	return render_template('term_utils.html', urls=session['apps_to_show'])

@app.route('/term_utils/search')
def term_utils_search():
	acct = request.form['account'].strip()
	proj = request.form['project'].strip()
	query = request.form['query'].strip()
	cli = LuminosoClient.connect('/projects/'+acct+'/'+proj,
							username=session['username'],
							password=session['password'])
	return jsonify(search_terms(query, cli))

@app.route('/term_utils/merge')
def term_utils_merge():
	###code
	return render_template('term_utils.html', urls=session['apps_to_show'])

@app.route('/term_utils/ignore')
def term_utils_ignore():
	###code
	return render_template('term_utils.html', urls=session['apps_to_show'])

if __name__ == '__main__':
    app.run(debug=True)















