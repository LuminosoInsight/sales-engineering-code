from flask import Flask, jsonify, render_template, request, session, url_for, Response
from luminoso_api import V5LuminosoClient
from pack64 import unpack64
from topic_utilities import copy_topics, del_topics, parse_url
from term_utilities import get_terms, ignore_terms, merge_terms
from rd_utilities import search_subsets
from deduper_utilities import dedupe
import numpy as np
from boilerplate_utilities import BPDetector, boilerplate_create_proj
from qualtrics_utilities import *
import redis
from se_code.conjunctions_disjunctions import get_new_results, get_current_results
from text_filter import filter_project
from subset_filter import filter_subsets
from auto_plutchik import get_all_topics, delete_all_topics, add_plutchik, copy_project
from compass_utilities import get_all_docs
from random import randint
from se_code.tableau_export import pull_lumi_data, create_doc_table, create_doc_term_table, create_doc_topic_table, create_doc_subset_table, create_themes_table, create_skt_table, create_drivers_table, write_table_to_csv, create_terms_table

#Storage for live classifier demo
classifiers = None
vectorizers = None
train_client = None
results = []

#Implement this for login checking for each route http://flask.pocoo.org/snippets/8/

app = Flask(__name__)
app.secret_key = 'secret_key_that_we_need_to_have_to_use_sessions'
red = redis.StrictRedis()

def connect_to_client(url):
    api_url, from_proj = parse_url(url)
    
    client = V5LuminosoClient.connect_with_username_and_password(url=api_url,
                                                               username=session['username'],
                                                               password=session['password'])
    client = client_for_path('projects/{}'.format(from_proj))
    return client

@app.route('/')
def home():
    return render_template('login.html')

@app.route('/', methods=['GET', 'POST'])
def login():
    session['username'] = request.form['username']
    session['password'] = request.form['password']
    session['apps_to_show'] = [
        ('Topic',('Copy Topics',url_for('copy_topics_page')),('Delete Topics',url_for('delete_topics_page'))),
        ('Term',('Merge Terms',url_for('term_merge_page')),('Ignore Terms',url_for('term_ignore_page'))),
        #('Cleaning',('Deduper',url_for('deduper_page')), ('Boilerplate Cleaner',url_for('boilerplate_page'))),
        ('Import/Export',('Qualtrics Survey Export',url_for('qualtrics'))),
        ('R&D Code',('Conjunction/Disjunction',url_for('conj_disj')),('Conceptual Subset Search',url_for('subset_search'))),
        ('Modify', ('Text Filter', url_for('text_filter_page')), ('Auto Emotions', url_for('plutchik_page')), ('Subset Filter', url_for('subset_filter_page'))),
        ('Dashboards', ('Tableau Export',url_for('tableau_export_page')))]
    try:
        V5LuminosoClient.connect_with_username_and_password('/projects', username=session['username'],
                                                                       password=session['password'])

        return render_template('welcome.html', urls=session['apps_to_show'])
    except Exception as e:
        print(e)
        error = 'Invalid_credentials'
        return render_template('login.html', error=error)

@app.route('/index')
def index():
    return render_template('index.html', urls=session['apps_to_show'])
    
@app.route('/tableau_export_page', methods=['GET'])
def tableau_export_page():
    return render_template('tableau_export.html', urls=session['apps_to_show'])

@app.route('/tableau_export', methods=['POST'])
def tableau_export():
    url = request.form['url'].strip()
    api_url, proj = parse_url(url)
    foldername = request.form['folder_name'].strip()
    concept_count = request.form['term_count'].strip()
    if concept_count == '':
        concept_count = 100
    else:
        concept_count = int(concept_count)
    skt_limit = request.form['skt_limit'].strip()
    if skt_limit == '':
        skt_limit = 20
    else:
        skt_limit = int(skt_limit)
        
    term_table = (request.form.get('terms') == 'on')
    doc_term = (request.form.get('doc_term') == 'on')
    doc_topic = (request.form.get('doc_topic') == 'on')
    doc_subset = (request.form.get('doc_subset') == 'on')
    themes_on = (request.form.get('themes') == 'on')
    skt_on = (request.form.get('skt') == 'on')
    drivers_on = (request.form.get('drivers') == 'on')
    #trends = (request.form.get('trends') == 'on')
    topic_drive = (request.form.get('topic_drive') == 'on')
    
    
    client, docs, saved_concepts, concepts, metadata, driver_fields, skt, themes = pull_lumi_data(proj, api_url, skt_limit=int(args.skt_limit), concept_count=int(args.concept_count))

    doc_table, xref_table, metadata_map = create_doc_table(client, docs, metadata)
    write_table_to_csv(doc_table, foldername, 'doc_table.csv')
    write_table_to_csv(xref_table, foldername, 'xref_table.csv')
    
    if term_table:
        terms_table = create_terms_table(concepts)
        write_table_to_csv(terms_table, foldername, 'terms_table.csv')
    if doc_term:
        doc_term_table = create_doc_term_table(docs, concepts)
        write_table_to_csv(doc_term_table, foldername, 'doc_term_table.csv')
    
    if doc_topic:
        doc_topic_table = create_doc_topic_table(docs, saved_concepts)
        write_table_to_csv(doc_topic_table, foldername, 'doc_topic_table.csv')
        
    if doc_subset:
        doc_subset_table = create_doc_subset_table(docs, metadata_map)
        write_table_to_csv(doc_subset_table, foldername, 'doc_subset_table.csv')
    
    if themes_on:
        themes_table = create_themes_table(client, themes)
        write_table_to_csv(themes_table, foldername, 'themes_table.csv')

    if skt_on:
        skt_table = create_skt_table(client, skt)
        write_table_to_csv(skt_table, foldername, 'skt_table.csv')
    
    if drivers_on:
        driver_table = create_drivers_table(client, driver_fields, topic_drive)
        write_table_to_csv(driver_table, foldername, 'drivers_table.csv')
    
    #if trends:
    #    trends_table, trendingterms_table = create_trends_table(terms, topics, docs)
    #    write_table_to_csv(trends_table, foldername, 'trends_table.csv')
    #    write_table_to_csv(trendingterms_table, foldername, 'trendingterms_table.csv')
    
    return render_template('tableau_export.html', urls=session['apps_to_show'])

@app.route('/plutchik', methods=['POST'])
def plutchik():
    url = request.form['url'].strip()
    client = connect_to_client(url)

    delete = (request.form.get('delete') == 'on')
    name = request.form['dest_name'].strip()
    copy = (request.form.get('copy') == 'on')
    to_acct = request.form['dest_acct'].strip()
    if to_acct == '':
        to_acct = from_acct
    
    topic_list = get_all_topics(client)
    if copy:
        client = copy_project(client, to_acct, name)
    if delete:
        delete_all_topics(client, topic_list)
    add_plutchik(client)
    return render_template('auto_plutchik.html', urls=session['apps_to_show'])
    

@app.route('/plutchik_page')
def plutchik_page():
    return render_template('auto_plutchik.html', urls=session['apps_to_show'])

@app.route('/subset_search', methods=['GET','POST'])
def subset_search():
    
    global client, subset_list, subset_vecs, field
    
    if request.method == 'POST':
        if 'url' in request.form:
            url = request.form['url'].strip()
            field = request.form['field'].strip()
            client = connect_to_client(url)
            project = client.get()['name']
            if field:
                docs = get_all_docs(client)
                subset_list = {}
                for doc in docs:
                    for metadata in doc['metadata']:
                        if metadata['name'] == field:
                            if metadata['value'] in subset_list:
                                subset_list[metadata['value']]['count'] += 1
                                subset_list[metadata['value']]['vector'] = np.sum([[float(v) for v in unpack64(doc['vector'])],
                                                                                   subset_list[metadata['value']]['vector']],
                                                                                   axis=0)
                            else:
                                subset_list[metadata['value']] = {'count': 1,
                                                                  'vector': [float(v) for v in unpack64(doc['vector'])],
                                                                  'subset': metadata['value']}
                            break
        else:
            project = client.get()['name']
            question = request.form['text']
            query_info, results = search_subsets(client,
                                                 question,
                                                 subset_vecs,
                                                 subset_list,
                                                 top_reviews=2,
                                                 field=field)
            return render_template('subset_search.html',
                                   urls=session['apps_to_show'],
                                   query_info=query_info,
                                   results=results,
                                   project=project)
    else:
        project = ''
        
    return render_template('subset_search.html', urls=session['apps_to_show'], project=project)

@app.route('/conj_disj', methods=['POST','GET'])
def conj_disj():
    new_results = []
    current_results = []
    query_info = ''
    
    if request.method == 'POST':
        url = request.form['url'].strip()
        client = connect_to_client(url)
                    
        new_results = get_new_results(client,
                                  request.form['search_terms'].split(','),
                                  request.form['unit'],
                                  int(request.form['n']),
                                  request.form['operation'])
        
        current_results = get_current_results(client,
                                  request.form['search_terms'].split(','),
                                  request.form['unit'],
                                  int(request.form['n']))
        
        connector = ' AND '
        if request.form['operation'] == 'disjunction':
            connector = ' OR '
            
        query_info = 'Results for {}'.format(connector.join(request.form['search_terms'].split(',')))
    
    return render_template('conj_disj.html',
                           urls=session['apps_to_show'],
                           results=list(zip(new_results, current_results)),
                           query_info=query_info)

@app.route('/topic_utils')
def topic_utils():
    return render_template('topic_utils.html', urls=session['apps_to_show'])

@app.route('/topic_utils/copy', methods=['POST'])
def topic_utils_copy():
    #NOTE: Should add a checkbox for if the existing topics should be deleted first
    url = request.form['url'].strip()
    api_url, from_proj = parse_url(url)
    client = connect_to_client(url)
    client = client.client_for_path('/')
    client = client.client_for_path('projects')
    dests = [url.strip() for url in request.form['dest_urls'].split(',')]

    for dest_proj in dests:
        api_url, to_proj = parse_url(dest_proj)
        copy_topics(
            client,
            from_proj=from_proj,
            to_proj=to_proj
        )
    #NOTE: ADD A FLASH CONFIRMATION MESSAGE HERE
    return render_template('copy_topics.html', urls=session['apps_to_show'])

@app.route('/topic_utils/delete', methods=['POST'])
def topic_utils_delete():
    dests = [url.strip() for url in request.form['urls'].split(',')]

    for dest_proj in dests:
        client = connect_to_client(dest_proj)
        del_topics(client)
    #NOTE: ADD A FLASH CONFIRMATION MESSAGE HERE
    return render_template('delete_topics.html', urls=session['apps_to_show'])

@app.route('/text_filter', methods=['POST'])
def text_filter():
    url = request.form['url'].strip()
    client = connect_to_client(url)
    text = request.form['remove'].strip()
    exact = (request.form.get('exact') == 'on')
    name = request.form['dest_name'].strip()
    filter_project(client=client, name=name, text=text, exact=exact)
    return render_template('text_filter.html', urls=session['apps_to_show'])
    

@app.route('/subset_filter', methods=['POST'])
def subset_filter():
    url = request.form['url'].strip()
    count = int(request.form['min_count'].strip())
    name = request.form['dest_name'].strip()
    more = (request.form.get('more') == 'on')
    subset_name = request.form['subset_name'].strip()
    only = (request.form.get('only') == 'on')
    filter_subsets(url=url, proj_name=name, subset_name=subset_name, count=count, 
                   only=only, more=more)
    return render_template('subset_filter.html', urls=session['apps_to_show'])

@app.route('/subset_filter_page')
def subset_filter_page():
    return render_template('subset_filter.html', urls=session['apps_to_show'])
    
@app.route('/text_filter_page')
def text_filter_page():
    return render_template('text_filter.html', urls=session['apps_to_show'])

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
    cli = connect_to_client(url)
    return jsonify(get_terms(cli))

@app.route('/term_utils/merge')
def term_utils_merge():
    url = request.args.get('url', 0, type=str)
    terms = eval(request.args.get('terms', 0, type=str))
    cli = connect_to_client(url)
    return jsonify(merge_terms(cli, terms))

@app.route('/term_utils/ignore')
def term_utils_ignore():
    url = request.args.get('url', 0, type=str).strip()
    terms = eval(request.args.get('terms', 0, type=str))
    cli = connect_to_client(url)
    return jsonify(ignore_terms(cli, terms))

#@app.route('/deduper_page')
#def deduper_page():
#    return render_template('dedupe.html', urls=session['apps_to_show'])

#@app.route('/dedupe')
#def dedupe_util():
#    url = request.args.get('url', 0, type=str)
#    api_url, acct, proj = parse_url(url)
#    copy = (request.args.get('copy') == 'true')
#    print(copy)
#    recalc = (request.args.get('recalc') == 'true')
#    reconcile = request.args.get('reconcile')
#    cli = LuminosoClient.connect('{}/projects/{}/{}'.format(api_url, acct, proj),
#                                 username=session['username'],
#                                 password=session['password'])
#    return jsonify(dedupe(acct=acct, proj=proj, cli=cli,
#            recalc=recalc, reconcile_func=reconcile, copy=copy))

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

def event_stream():
    pubsub = red.pubsub()
    pubsub.subscribe('boilerplate')
    for message in pubsub.listen():
        if message['type'] == 'subscribe':
            pass
        else:
            yield str('data: %s\n\n' % message['data'])

#@app.route('/boilerplate_page')
#def boilerplate_page():
#    return render_template('boilerplate.html', urls=session['apps_to_show'])

#@app.route('/boilerplate_stream')
#def boilerplate_stream():
#    return Response(event_stream(), content_type='text/event-stream')

#@app.route('/boilerplate_run')
#def bp_run():
#    thresh = request.args.get('thresh', 6, type=int)
#    window_size = request.args.get('window_size', 7,  type=int)
#    use_gaps = request.args.get('use_gaps', "on", type=str)
#    sample_docs = request.args.get('sample_docs', 10, type=int)
#    url = request.args.get('url', type=str)
#    api_url, acct, proj = parse_url(url)
#    bp = BPDetector()
#    bp.threshold = thresh
#    bp.window_size = window_size
#    bp.use_gaps = use_gaps == "on"
#    output_fp, name, acct = bp.run(api_url=api_url, acct=acct, proj=proj,
#            user=session['username'],
#            passwd=session['password'],
#            sample_docs=sample_docs,
#            redis=red,
#            train=True,
#            tokens_to_scan=1000000,
#            verbose=True)
#    return jsonify({'output': output_fp, 'name': name, 'acct': acct})

#@app.route('/boilerplate_new_proj')
#def bp_create_proj():
#    filepath = request.args.get('docs_path', type=str)
#    name = request.args.get('name', type=str)
#    acct = request.args.get('acct', type=str)
#    recalc = (request.args.get('recalc') == 'true')
#    acct, proj = boilerplate_create_proj(filepath, name, acct, recalc,
#                            username = session['username'],
#                            password = session['password'])
#    url = 'https://analytics.luminoso.com/explore.html?account='+acct+'&projectId='+proj
#    return jsonify({'proj_url': url})



###
# END Boilerplate code
###

if __name__ == '__main__':
    app.run()#, ssl_context='adhoc')



















