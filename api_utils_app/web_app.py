from flask import Flask, jsonify, render_template, request, session, url_for, Response, stream_with_context
import json
from collections import defaultdict, OrderedDict
from luminoso_api import LuminosoClient
from topic_utilities import copy_topics, del_topics, parse_url
from term_utilities import get_terms, ignore_terms, merge_terms
from deduper_utilities import dedupe
from boilerplate_utilities import BPDetector
from qualtrics_utilities import *
import itertools
import redis
#from boilerplate_utilities import run

app = Flask(__name__)
app.secret_key = 'secret_key_that_we_need_to_have_to_use_sessions'
red = redis.StrictRedis()

@app.route('/')
def home():
    return render_template('login.html')

@app.route('/', methods=['GET', 'POST'])
def login():
    session['username'] = request.form['username']
    session['password'] = request.form['password']
    session['apps_to_show'] = [('Topic',('Copy Topics',url_for('copy_topics_page')),('Delete Topics',url_for('delete_topics_page'))),
                            ('Term',('Merge Terms',url_for('term_merge_page')),('Ignore Terms',url_for('term_ignore_page'))),
                            ('Cleaning',('Deduper',url_for('deduper_page')), ('Boilerplate Cleaner',url_for('boilerplate_page'))),
                            ('CSV Exports',('Compass Messages Export',url_for('compass_export_page')),('Analytics Docs Export',url_for('compass_export_page'))),
                            ('Import/Export',('Qualtrics Survey Export',url_for('qualtrics')))]
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
    url = request.form['url'].strip()
    from_acct, from_proj = parse_url(url)
    dests = [parse_url(url.strip()) for url in request.form['dest_urls'].split(',')]
    cli = LuminosoClient.connect('/projects/', username=session['username'],
                                            password=session['password'])
    for dest_proj in dests:
        to_acct, to_proj = dest_proj
        copy_topics(cli, from_acct=from_acct, from_proj=from_proj,
                        to_acct=to_acct, to_proj=to_proj)
    #NOTE: ADD A FLASH CONFIRMATION MESSAGE HERE
    return render_template('copy_topics.html', urls=session['apps_to_show'])

@app.route('/topic_utils/delete', methods=['POST'])
def topic_utils_delete():
    dests = [parse_url(url.strip()) for url in request.form['urls'].split(',')]
    cli = LuminosoClient.connect('/projects/', username=session['username'],
                                            password=session['password'])
    for dest_proj in dests:
        acct, proj = dest_proj
        del_topics(cli, acct_id=acct, proj_id=proj)
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
    url = request.args.get('url', 0, type=str).strip()
    acct, proj = parse_url(url)
    cli = LuminosoClient.connect('/projects/'+acct+'/'+proj,
                            username=session['username'],
                            password=session['password'])
    return jsonify(get_terms(cli))

@app.route('/term_utils/merge')
def term_utils_merge():
    url = request.args.get('url', 0, type=str)
    acct, proj = parse_url(url)
    terms = eval(request.args.get('terms', 0, type=str))
    cli = LuminosoClient.connect('/projects/'+acct+'/'+proj,
                            username=session['username'],
                            password=session['password'])
    return jsonify(merge_terms(cli, terms))

@app.route('/term_utils/ignore')
def term_utils_ignore():
    url = request.args.get('url', 0, type=str).strip()
    acct, proj = parse_url(url)
    terms = eval(request.args.get('terms', 0, type=str))
    cli = LuminosoClient.connect('/projects/'+acct+'/'+proj,
                            username=session['username'],
                            password=session['password'])
    return jsonify(ignore_terms(cli, terms))

@app.route('/deduper_page')
def deduper_page():
    return render_template('dedupe.html', urls=session['apps_to_show'])

@app.route('/dedupe')
def dedupe_util():
    url = request.args.get('url', 0, type=str)
    acct, proj = parse_url(url)
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
                                       user=session['username'],
                                       passwd=session['password'],
                                       name=title+" (Imported from Qualtrics)")
    return jsonify({"url":proj_url})

###
# BEGIN Boilerplate code, some of which will be moved to separate file
# note: may need to use the session['username'] to uniquely identify
#       the pubsub messages for each user, if red is shared for all sessions.
###

###TODO: pass red to the bp.run() and then have the sample docs published to red.

def event_stream():
    pubsub = red.pubsub()
    pubsub.subscribe('boilerplate')
    # TODO: handle client disconnection.
    for message in pubsub.listen():
        if message['type'] == 'subscribe':
            pass
        else:    
            yield str('data: %s\n\n' % message['data'])

@app.route('/boilerplate_page')
def boilerplate_page():
    return render_template('boilerplate.html', urls=session['apps_to_show'])

@app.route('/boilerplate_stream')
def boilerplate_stream():
    return Response(event_stream(), content_type='text/event-stream')

@app.route('/boilerplate_run')
def bp_run():
    thresh = request.args.get('thresh', 6, type=int)
    window_size = request.args.get('window_size', 7,  type=int)
    use_gaps = request.args.get('use_gaps', "on", type=str)
    sample_docs = request.args.get('sample_docs', 10, type=int)
    url = request.args.get('url', type=str)
    print(thresh, window_size, use_gaps, sample_docs, url)
    bp = BPDetector()
    bp.threshold = thresh
    bp.window_size = window_size
    bp.use_gaps = use_gaps == "on"
    bp.run(input='/users/tobrien/Desktop/archive/staples_transcripts_10k.json',
            output='/users/tobrien/Desktop/webapp_bp_test.json',
            output_ngrams='/users/tobrien/Desktop/webapp_bp_test_ngrams.json',
            sample_docs=sample_docs,
            redis=red,
            train=True,
            tokens_to_scan=1000000,
            verbose=True)
    return jsonify({'status': 'finished'})

###
# END Boilerplate code
###

if __name__ == '__main__':
    app.run(debug=True, threaded=True, host= '0.0.0.0')






