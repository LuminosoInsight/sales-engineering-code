from flask import Flask, jsonify, render_template, request, session, url_for
import json
from collections import defaultdict, OrderedDict
from luminoso_api import LuminosoClient
from topic_utilities import copy_topics, del_topics
from term_utilities import search_terms, ignore_terms, merge_terms
from deduper_utilities import dedupe
from qualtrics_utilities import *

app = Flask(__name__)
app.secret_key = 'secret_key_that_we_need_to_have_to_use_sessions'

@app.route('/')
def home():
    return render_template('login.html')

@app.route('/', methods=['GET', 'POST'])
def login():
	session['username'] = request.form['username']
	session['password'] = request.form['password']
	session['apps_to_show'] = [('Topic',('Copy Topics',url_for('copy_topics_page')),('Delete Topics',url_for('delete_topics_page'))),
           					('Term',('Merge Terms',url_for('term_merge_page')),('Ignore Terms',url_for('term_ignore_page'))),
           					('Cleaning',('Deduper',url_for('deduper_page')),('Cleaning2',url_for('deduper_page'))),
           					('CSV Exports',('Compass Messages Export',url_for('compass_export_page')),('Analytics Docs Export',url_for('deduper_page'))),
           					('Import/Export',('Qualtrics',url_for('qualtrics')),('Compass -> Analytics',url_for('qualtrics')))]
	print(session['apps_to_show'])
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
	return render_template('copy_topics.html', urls=session['apps_to_show'])


@app.route('/topic_utils/delete', methods=['POST'])
def topic_utils_delete():
	acct = request.form['account'].strip()
	dests = [p_id.strip() for p_id in request.form['pids'].split(',')]
	cli = LuminosoClient.connect('/projects/', username=session['username'],
									   		password=session['password'])
	for dest_proj in dests:
		del_topics(cli, acct_id=acct, proj_id=dest_proj)
	#NOTE: ADD A FLASH CONFIRMATION MESSAGE HERE
	return render_template('delete_topics.html', urls=session['apps_to_show'])

@app.route('/term_utils')
def term_utils():
	return render_template('term_utils.html', urls=session['apps_to_show'])

@app.route('/term_merge_page')
def term_merge_page():
	return render_template('term_merge.html', urls=session['apps_to_show'])

@app.route('/term_ignore')
def term_ignore_page():
	return render_template('term_ignore.html', urls=session['apps_to_show'])

@app.route('/copy_topics_page')
def copy_topics_page():
	return render_template('copy_topics.html', urls=session['apps_to_show'])

@app.route('/delete_topics_page')
def delete_topics_page():
	return render_template('delete_topics.html', urls=session['apps_to_show'])

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

@app.route('/compass_export_page')
def compass_export_page():
	return render_template('compass_export.html', urls=session['apps_to_show'])

@app.route('/compass_export')
def compass_export():
	proj = request.args.get('proj', 0, type=str)
	staging = (request.args.get('staging') == 'true')
	spams = (request.args.get('spams') == 'true')
	#code here
	results={"hello":"testing"}
	return jsonify(results)

# Qualtrics routes
@app.route('/qualtrics')
def qualtrics():
    return render_template('qual_simple.html', urls=session['apps_to_show'])

@app.route('/qualtrics_step1')
def step1():
    """From the given token, get name and id of surveys"""
    token = request.args.get('token', 0, type=str)
    info = get_name_id(token) #returns a dictionary
    return jsonify(**info)

@app.route('/qualtrics_step2')
def step2():
    """Given sid and token, return the question info"""
    sid = request.args.get('sid', 0, type=str)
    token = request.args.get('token', 0, type=str)
    return jsonify(get_question_descriptions(sid, token))

@app.route('/qualtrics_step3')
def step3():
    """Given sid, token, text question ID, subset question IDs,
       a luminoso account and token, create a proj in Analytics"""
    sid = request.args.get('sid', 0, type=str)
    token = request.args.get('token', 0, type=str)
    text_qs = eval(request.args.get('text_qs', 0, type=str))
    subset_qs = eval(request.args.get('subset_qs', 0, type=str))
    title = request.args.get('title', 0, type=str)
    proj_url = build_analytics_project(sid, token, text_qs, subset_qs,
                                       lumi_account, lumi_token, title+" (Imported from Qualtrics)")
    return jsonify({"url":proj_url})

if __name__ == '__main__':
    app.run(debug=True, threaded=True)