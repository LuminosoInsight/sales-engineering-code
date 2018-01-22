from flask import Flask, jsonify, render_template, request, session, url_for, Response
from luminoso_api import LuminosoClient
from pack64 import unpack64
from topic_utilities import copy_topics, del_topics, parse_url
from se_code.run_voting_classifier import return_label, train_classifier, get_docs_labels, split_train_test
from term_utilities import get_terms, ignore_terms, merge_terms
from rd_utilities import search_subsets
from deduper_utilities import dedupe
import numpy as np
from boilerplate_utilities import BPDetector, boilerplate_create_proj
from qualtrics_utilities import *
import redis
from conjunction_disjunction import get_new_results, get_current_results
from text_filter import filter_project
from subset_filter import filter_subsets
from auto_plutchik import get_all_topics, delete_all_topics, add_plutchik, copy_project
from compass_utilities import post_messages, format_messages, get_all_docs
from random import randint
from tableau_export_web import reorder_subsets, pull_lumi_data, create_doc_table, create_doc_term_table, create_doc_topic_table, create_doc_subset_table, create_themes_table, create_skt_table, create_drivers_table, create_trends_table, write_table_to_csv, create_terms_table

#Storage for live classifier demo
classifiers = None
vectorizers = None
train_client = None
results = []

#Implement this for login checking for each route http://flask.pocoo.org/snippets/8/

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
    session['apps_to_show'] = [
        ('Topic',('Copy Topics',url_for('copy_topics_page')),('Delete Topics',url_for('delete_topics_page'))),
        ('Term',('Merge Terms',url_for('term_merge_page')),('Ignore Terms',url_for('term_ignore_page'))),
        ('Cleaning',('Deduper',url_for('deduper_page')), ('Boilerplate Cleaner',url_for('boilerplate_page'))),
        ('CSV Exports',('Compass Messages Export',url_for('compass_export_page')),('Analytics Docs Export',url_for('compass_export_page'))),
        ('Import/Export',('Qualtrics Survey Export',url_for('qualtrics'))),
        ('R&D Code',('Conjunction/Disjunction',url_for('conj_disj')),('Conceptual Subset Search',url_for('subset_search'))),
        ('Classification',('Setup Voting Classifier Demo',url_for('classifier_demo')), ('Compass Demo',url_for('compass_demo'))),
        ('Modify', ('Text Filter', url_for('text_filter_page')), ('Auto Emotions', url_for('plutchik_page')), ('Subset Filter', url_for('subset_filter_page'))),
        ('Dashboards', ('Tableau Export',url_for('tableau_export_page')))]
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

@app.route('/compass_demo', methods=['GET'])
def compass_demo():
    return render_template('compass_demo.html', urls=session['apps_to_show'])

@app.route('/compass_stream', methods=['POST'])
def compass_stream():
    url = request.form['url'].strip()
    from_acct, from_proj = parse_url(url)
    client = LuminosoClient.connect('/projects/', username=session['username'],
                                               password=session['password'])
    api_url = request.form['api_url']
    client = LuminosoClient.connect('/projects/{}/{}'.format(from_acct, from_proj))
    docs = get_all_docs(client)
    avg_doc_len = np.mean([len(d['text']) for d in docs[:100]])
    if 'date' in docs[0]: #assumes all docs have a date if the first one does
        docs = sorted(docs, key=lambda k: k['date'])
    compass_username = request.form['comp_name']
    compass_password = request.form['comp_pass']
    stream_time = request.form['stream_time']
    total_time = 0
    slice_start = 0
    while total_time < int(float(stream_time) * 60):
        batch_size = randint(1, min(int(40000/avg_doc_len), int(len(docs) / 10)))
        interval = randint(int(batch_size / 10), int(batch_size / 5))
        curr_docs = docs[slice_start:slice_start+batch_size]
        slice_start += batch_size
        messages = format_messages(curr_docs)
        post_messages(api_url, messages, interval, compass_username, compass_password)
        print('POSTed {}, sleeping for {}'.format(batch_size, interval))
        total_time += max(interval, 1)
    print('Done posting')
    return render_template('compass_demo.html', urls=session['apps_to_show'])
    
@app.route('/tableau_export_page', methods=['GET'])
def tableau_export_page():
    return render_template('tableau_export.html', urls=session['apps_to_show'])

@app.route('/tableau_export', methods=['POST'])
def tableau_export():
    url = request.form['url'].strip()
    from_acct, from_proj = parse_url(url)
    foldername = request.form['folder_name'].strip()
    term_count = request.form['term_count'].strip()
    if term_count == '':
        term_count = 100
    else:
        term_count = int(term_count)
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
    trends = (request.form.get('trends') == 'on')
    driver_rebuild = (request.form.get('rebuild') == 'on')
    topic_drive = (request.form.get('topic_drive') == 'on')
    average_score = (request.form.get('average_score') == 'on')
    
    client, docs, topics, terms, subsets, drivers, skt, themes = pull_lumi_data(from_acct, from_proj, skt_limit=skt_limit, term_count=term_count, rebuild=driver_rebuild)
    subsets = reorder_subsets(subsets)

    doc_table, xref_table = create_doc_table(client, docs, subsets, themes, drivers)
    write_table_to_csv(doc_table, foldername, 'doc_table.csv')
    write_table_to_csv(xref_table, foldername, 'xref_table.csv')
    
    if term_table:
        terms_table = create_terms_table(client, terms)
        write_table_to_csv(terms_table, foldername, 'terms_table.csv')
    if doc_term:
        doc_term_table = create_doc_term_table(client, docs, terms, .3)
        write_table_to_csv(doc_term_table, foldername, 'doc_term_table.csv')
    
    if doc_topic:
        doc_topic_table = create_doc_topic_table(client, docs, topics)
        write_table_to_csv(doc_topic_table, foldername, 'doc_topic_table.csv')
        
    if doc_subset:
        doc_subset_table = create_doc_subset_table(client, docs, subsets)
        write_table_to_csv(doc_subset_table, foldername, 'doc_subset_table.csv')
    
    if themes_on:
        themes_table = create_themes_table(client, themes)
        write_table_to_csv(themes_table, foldername, 'themes_table.csv')

    if skt_on:
        skt_table = create_skt_table(client, skt)
        write_table_to_csv(skt_table, foldername, 'skt_table.csv')
    
    if drivers_on:
        driver_table = create_drivers_table(client, drivers, topic_drive, average_score)
        write_table_to_csv(driver_table, foldername, 'drivers_table.csv')
    
    if trends:
        trends_table, trendingterms_table = create_trends_table(terms, topics, docs)
        write_table_to_csv(trends_table, foldername, 'trends_table.csv')
        write_table_to_csv(trendingterms_table, foldername, 'trendingterms_table.csv')
    
    return render_template('tableau_export.html', urls=session['apps_to_show'])

@app.route('/classifier_demo', methods=['GET'])
def classifier_demo():
    return render_template('setup_classifier.html', urls=session['apps_to_show'])

@app.route('/classifier_demo', methods=['POST'])
def setup_classifier():
    global classifiers
    global vectorizers
    global train_client
    global results
    
    if request.method == 'POST':
        train_url = request.form['train_url'].strip()
        train_acct, training_project_id = parse_url(train_url)
        test_url = request.form['test_url'].strip()
        test_acct, testing_project_id = parse_url(test_url)
        subset_field = request.form['subset_label'].strip()
        
        client = LuminosoClient.connect(username=session['username'],
                                        password=session['password'])
        
        train_client = client.change_path('/projects/{}/{}'.format(train_acct,training_project_id))
        
        if training_project_id == testing_project_id:
            docs, labels = get_docs_labels(train_client, subset_field)
            train_docs, test_docs, train_labels, test_labels = split_train_test(docs, labels)
        else:
            test_client = client.change_path('/projects/{}/{}'.format(test_acct, testing_project_id))
            train_docs, train_labels = get_docs_labels(train_client, subset_field)
            test_docs, test_labels = get_docs_labels(test_client, subset_field)
        
        classifiers, vectorizers = train_classifier(
            train_docs, train_labels
            )
        
        sample_results = [doc['text'] for doc in np.random.choice(test_docs,100)]
        global results 
        results = []
        for sample in sample_results:
            result = list(return_label(sample, classifiers, vectorizers, train_client))
            result.append(sample)
            results.append(result)
            
    return render_template('classifier.html', urls=session['apps_to_show'], results=results)


@app.route('/live_classifier', methods=['GET','POST'])
def live_classifier():
    global results
    
    if results is None:
        return render_template('classifier.html', urls=session['apps_to_show'], results=[['','',0]])
    
    if request.method == 'POST':
    
        new_text = request.form['text']
        result = list(return_label(new_text, classifiers, vectorizers, train_client))
        result.append(new_text)
        results.append(result)

    return render_template('classifier.html', urls=session['apps_to_show'], results=results[::-1])

@app.route('/plutchik', methods=['POST'])
def plutchik():
    url = request.form['url'].strip()
    from_acct, from_proj = parse_url(url)
    client = LuminosoClient.connect('/projects/', username=session['username'],
                                                password=session['password'])
    client = client.change_path('/')
    client = client.change_path('/projects/{}/{}'.format(from_acct, from_proj))
    delete = (request.form.get('delete') == 'on')
    name = request.form['dest_name'].strip()
    copy = (request.form.get('copy') == 'on')
    
    topic_list = get_all_topics(client)
    if copy:
        client = copy_project(client, from_acct, name)
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
            from_acct, from_proj = parse_url(url)
            client = LuminosoClient.connect('/projects/{}/{}'.format(from_acct,from_proj),
                                            username=session['username'],
                                            password=session['password'])
            project = client.get()['name']
            if field:
                docs = get_all_docs(client)
                subset_list = {}
                for doc in docs:
                    if field in doc['source']:
                        subset = doc['source'][field]
                        if doc['source'][field] in subset_list:
                            subset_list[subset]['count'] += 1
                            subset_list[subset]['vector'] = np.sum([unpack64(doc['vector']),
                                subset_list[subset]['vector']], axis=0)
                        else:
                            subset_list[subset] = {'count':1,
                                         'vector':unpack64(doc['vector']),
                                         'subset':subset}
                #print(subset_list)
                subset_vecs = [s['vector']/s['count'] for k,s in subset_list.items()]
                subset_names = subset_list.keys()
                subset_list = list(subset_list.values())         
            else:
                subset_list = client.get('/subsets/stats')
                subset_vecs = [unpack64(s['mean']) for s in subset_list]
                subset_names = [s['subset'] for s in subset_list]
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
        from_acct, from_proj = parse_url(url)
        client = LuminosoClient.connect('/projects/{}/{}'.format(from_acct,from_proj), username=session['username'],
                                            password=session['password'])
        neg_terms = []
        if len(request.form['neg_terms']) > 0:
            neg_terms = request.form['neg_terms'].split(',')
                    
        new_results = get_new_results(client,
                                  request.form['search_terms'].split(','),
                                  neg_terms,
                                  request.form['unit'],
                                  int(request.form['n']),
                                  request.form['operation'],
                                  False)
        
        current_results = get_current_results(client,
                                  request.form['search_terms'].split(','),
                                  neg_terms,
                                  '',
                                  request.form['unit'],
                                  int(request.form['n']),
                                  False)
        
        connector = ' AND '
        if request.form['operation'] == 'disjunction':
            connector = ' OR '
        
        suffix = ''
        if neg_terms:
            suffix = ' NOT {}'.format(' '.join(neg_terms))
            
        query_info = 'Results for {}{}'.format(connector.join(request.form['search_terms'].split(',')),suffix)
    
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

@app.route('/text_filter', methods=['POST'])
def text_filter():
    url = request.form['url'].strip()
    from_acct, from_proj = parse_url(url)
    client = LuminosoClient.connect('/projects/', username=session['username'],
                                               password=session['password'])
    text = request.form['remove'].strip()
    #exact = (request.form['exact'] == 'on')
    exact = (request.form.get('exact') == 'on')
    reconcile = (request.form.get('reconcile') == 'unrelated')
    name = request.form['dest_name'].strip()
    #branch = (request.form['branch'] == 'on')
    branch = (request.form.get('branch') == 'on')
    filter_project(client=client, acc_id=from_acct, proj_id=from_proj, branch_name=name, 
                   text=text, not_related=reconcile, branch=branch, exact=exact)
    return render_template('text_filter.html', urls=session['apps_to_show'])
    #return jsonify(filter_project(from_acct, from_proj, name, text, reconcile, branch, exact))
    

@app.route('/subset_filter', methods=['POST'])
def subset_filter():
    url = request.form['url'].strip()
    from_acct, from_proj = parse_url(url)
    client = LuminosoClient.connect('/projects/', username=session['username'],
                                               password=session['password'])
    count = int(request.form['min_count'].strip())
    name = request.form['dest_name'].strip()
    more = (request.form.get('more') == 'on')
    subset_name = request.form['subset_name'].strip()
    only = (request.form.get('only') == 'on')
    filter_subsets(client=client, account_id=from_acct, project_id=from_proj, 
                   proj_name=name, subset_name=subset_name, count=count, only=only, more=more)
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
    print(copy)
    recalc = (request.args.get('recalc') == 'true')
    reconcile = request.args.get('reconcile')
    cli = LuminosoClient.connect('/projects/'+acct+'/'+proj,
                            username=session['username'],
                            password=session['password'])
    return jsonify(dedupe(acct=acct, proj=proj, cli=cli,
            recalc=recalc, reconcile_func=reconcile, copy=copy))

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

def event_stream():
    pubsub = red.pubsub()
    pubsub.subscribe('boilerplate')
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
    acct, proj = parse_url(url)
    bp = BPDetector()
    bp.threshold = thresh
    bp.window_size = window_size
    bp.use_gaps = use_gaps == "on"
    output_fp, name, acct = bp.run(acct=acct, proj=proj,
            user=session['username'],
            passwd=session['password'],
            sample_docs=sample_docs,
            redis=red,
            train=True,
            tokens_to_scan=1000000,
            verbose=True)
    return jsonify({'output': output_fp, 'name': name, 'acct': acct})

@app.route('/boilerplate_new_proj')
def bp_create_proj():
    filepath = request.args.get('docs_path', type=str)
    name = request.args.get('name', type=str)
    acct = request.args.get('acct', type=str)
    recalc = (request.args.get('recalc') == 'true')
    acct, proj = boilerplate_create_proj(filepath, name, acct, recalc,
                            username = session['username'],
                            password = session['password'])
    url = 'https://analytics.luminoso.com/explore.html?account='+acct+'&projectId='+proj
    return jsonify({'proj_url': url})



###
# END Boilerplate code
###

if __name__ == '__main__':
    app.run()#, ssl_context='adhoc')



















