from flask import Flask, jsonify, render_template, request, session, url_for
import json
from collections import defaultdict, OrderedDict
from luminoso_api import LuminosoClient
from topic_utilities import copy_topics, del_topics
from term_utilities import search_terms, ignore_terms, merge_terms
from deduper_utilities import dedupe

app = Flask(__name__)
app.secret_key = 'secret_key_that_we_need_to_have_to_use_sessions'

@app.route('/')
def home():
    return render_template('login.html')

@app.route('/', methods=['GET', 'POST'])
def login():
	session['username'] = request.form['username']
	session['password'] = request.form['password']

	"""session['apps_to_show'] = [('Topic Utilities',url_for('topic_utils')),
							   ('Term Utilities',url_for('term_utils')),
							   ('Cleaning Utilities',url_for('deduper_page'))]"""

	session['apps_to_show'] = [('Topic',('Copy Topics',url_for('topic_utils')),('Delete Topics',url_for('topic_utils'))),
           					('Term',('Merge Terms',url_for('term_utils')),('Ignore Terms',url_for('term_utils'))),
           					('Cleaning',('Deduper',url_for('deduper_page')),('Cleaning2',url_for('deduper_page')))]
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

@app.route('/term_utils/search', methods=['GET','POST'])
def term_utils_search():
	acct = request.args.get('acct', 0, type=str)
	proj = request.args.get('proj', 0, type=str)
	query = request.args.get('query', 0, type=str)
	cli = LuminosoClient.connect('/projects/'+acct+'/'+proj,
							username=session['username'],
							password=session['password'])
	return jsonify(search_terms(query, cli))

@app.route('/term_utils/merge')
def term_utils_merge():
	print("sent to server")
	acct = request.args.get('acct', 0, type=str)
	proj = request.args.get('proj', 0, type=str)
	terms = eval(request.args.get('terms', 0, type=str))
	print(acct)
	print(proj)
	print(terms)
	cli = LuminosoClient.connect('/projects/'+acct+'/'+proj,
							username=session['username'],
							password=session['password'])
	return jsonify(merge_terms(cli, terms))

@app.route('/term_utils/ignore')
def term_utils_ignore():
	print("sent to server")
	acct = request.args.get('acct', 0, type=str)
	proj = request.args.get('proj', 0, type=str)
	terms = eval(request.args.get('terms', 0, type=str))
	print(acct)
	print(proj)
	print(terms)
	cli = LuminosoClient.connect('/projects/'+acct+'/'+proj,
							username=session['username'],
							password=session['password'])
	return jsonify(ignore_terms(cli, terms))

@app.route('/deduper_page')
def deduper_page():
	return render_template('dedupe.html', urls=session['apps_to_show'])

@app.route('/dedupe')
def dedupe_util():
	acct = request.args.get('acct', 0, type=str)
	proj = request.args.get('proj', 0, type=str)
	copy = (request.args.get('copy') == 'true')
	reconcile = request.args.get('reconcile')
	cli = LuminosoClient.connect('/projects/'+acct+'/'+proj,
							username=session['username'],
							password=session['password'])
	return jsonify(dedupe(acct=acct, proj=proj, cli=cli,
					reconcile_func=reconcile, copy=copy))

if __name__ == '__main__':
    app.run(debug=True, threaded=True)